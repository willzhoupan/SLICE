// Provide by Jacques CHEN (http://whchen.net/index.php/About.html)
// HTML file reference from ChatGLM-MNN （https://github.com/wangzhaode/ChatGLM-MNN)

#include "httplib.h"
#include "model.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <stdlib.h>
#include <string>
#include <mutex>
#include <condition_variable>
#include <chrono>

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

struct WebConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string webPath = "web"; // 网页文件路径
    std::string systemPrompt = "You are a helpful assistant.";
    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式
    int port = 8081; // 端口号
    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    int groupCnt = -1;
    bool enableEageServe = false; // 是否启用EageServe调度
    int longTaskThreshold = 128; // 长任务阈值
    float priorityDegradeThreshold = 1.5f; // 优先级降级阈值
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--system> <args>:            设置系统提示词(system prompt)" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<-w|--web> <args>:            网页文件的路径" << std::endl;
    std::cout << "<--port> <args>:              网页端口号" << std::endl;
    std::cout << "<--eageserve>:                启用EageServe调度器" << std::endl;
    std::cout << "<--long-threshold> <args>:    设置长任务阈值(token数)" << std::endl;
    std::cout << "<--degrade-threshold> <args>: 设置优先级降级阈值(倍数)" << std::endl;
}

void ParseArgs(int argc, char **argv, WebConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--low") {
            config.lowMemMode = true;
        } else if (sargv[i] == "--system") {
            config.systemPrompt = sargv[++i];
        } else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "-w" || sargv[i] == "--web") {
            config.webPath = sargv[++i];
        } else if (sargv[i] == "--port") {
            config.port = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--eageserve") {
            config.enableEageServe = true;
        } else if (sargv[i] == "--long-threshold") {
            config.longTaskThreshold = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--degrade-threshold") {
            config.priorityDegradeThreshold = atof(sargv[++i].c_str());
        } else {
            Usage();
            exit(-1);
        }
    }
}

struct ChatSession {
    fastllm::ChatMessages messages;
    std::string input = "";
    std::string output = "";
    int status = 0; // 0: 空闲 1: 结果生成好了 2: 已经写回了
    bool isRealtime = false; // 是否为实时任务
    std::mutex sessionMutex; // 会话级别的互斥锁
    std::condition_variable resultCV; // 用于通知结果可用的条件变量
    bool isProcessing = false; // 是否正在处理中
    std::chrono::steady_clock::time_point lastAccessTime; // 上次访问时间，用于清理过期会话
    int handleId = -1; // 当前请求的handleId，用于中断请求
    
    ChatSession() {
        lastAccessTime = std::chrono::steady_clock::now();
    }
};

std::map <std::string, ChatSession*> sessions;
std::mutex sessionsLocker;

// 会话清理线程，用于清理长时间未使用的会话
void sessionCleanupThread(std::map<std::string, ChatSession*>& sessions, std::mutex& sessionsLocker, bool& running) {
    const auto sessionTimeout = std::chrono::minutes(30); // 会话超时时间
    
    while (running) {
        std::this_thread::sleep_for(std::chrono::minutes(5)); // 每5分钟检查一次
        
        std::vector<std::string> sessionsToRemove;
        {
            std::lock_guard<std::mutex> lock(sessionsLocker);
            auto now = std::chrono::steady_clock::now();
            
            for (auto& pair : sessions) {
                try {
                    std::lock_guard<std::mutex> sessionLock(pair.second->sessionMutex);
                    if (now - pair.second->lastAccessTime > sessionTimeout && !pair.second->isProcessing) {
                        sessionsToRemove.push_back(pair.first);
                    }
                } catch (...) {
                    // 忽略锁定失败，下次再检查
                }
            }
        }
        
        // 删除过期会话
        if (!sessionsToRemove.empty()) {
            std::lock_guard<std::mutex> lock(sessionsLocker);
            for (const auto& uuid : sessionsToRemove) {
                try {
                    auto it = sessions.find(uuid);
                    if (it != sessions.end()) {
                        delete it->second;
                        sessions.erase(it);
                    }
                } catch (...) {
                    // 忽略删除过程中的错误
                }
            }
            std::cout << "已清理 " << sessionsToRemove.size() << " 个过期会话" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    WebConfig config;
    ParseArgs(argc, argv, config);

    try {
        fastllm::SetThreads(config.threads);
        fastllm::SetLowMemMode(config.lowMemMode);
        if (!fastllm::FileExists(config.path)) {
            printf("模型文件 %s 不存在！\n", config.path.c_str());
            exit(0);
        }
        bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
        auto model = isHFDir ? fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt)
            : fastllm::CreateLLMModelFromFile(config.path);

        // 配置EageServe
        if (config.enableEageServe) {
            model->enableEageServe = true;
            model->longTaskThreshold = config.longTaskThreshold;
            model->priorityDegradeThreshold = config.priorityDegradeThreshold;
            printf("EageServe调度器已启用，长任务阈值: %d tokens，优先级降级阈值: %.1f倍\n", 
                config.longTaskThreshold, config.priorityDegradeThreshold);
        }
        
        // 启动会话清理线程
        bool cleanupThreadRunning = true;
        std::thread cleanup(sessionCleanupThread, std::ref(sessions), std::ref(sessionsLocker), std::ref(cleanupThreadRunning));
        cleanup.detach();

        httplib::Server svr;
        auto chat = [&](ChatSession *session, const std::string input) {
            try {
                // 更新会话的最后访问时间
                session->lastAccessTime = std::chrono::steady_clock::now();
                
                if (input == "reset" || input == "stop") {
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                    if (!session->messages.empty()) {
                        session->messages.erase(std::next(session->messages.begin()), session->messages.end());
                    }
                    session->output = "<eop>\n";
                    session->status = 2;
                    session->isProcessing = false;
                    session->resultCV.notify_all();
                    return;
                }
                
                // 设置处理中标志
                {
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                    session->isProcessing = true;
                    session->messages.push_back(std::make_pair("user", input));
                }
                
                auto prompt = model->ApplyChatTemplate(session->messages);
                auto inputs = model->weight.tokenizer.Encode(prompt);

                std::vector<int> tokens;
                for (int i = 0; i < inputs.Count(0); i++) {
                    tokens.push_back(((float *) inputs.cpuData)[i]);
                }

                // 创建生成配置
                fastllm::GenerationConfig genConfig;
                
                // 根据任务类型设置不同的输出限制，防止长任务阻塞系统
                if (session->isRealtime) {
                    genConfig.output_token_limit = 50; // 实时任务限制输出token数
                } else {
                    genConfig.output_token_limit = 100; // 非实时任务可以生成更多token
                }
                
                // 优化提示语类型判断，根据内容长度和关键词判断
                bool isShortTask = input.length() < 50 && input.find("详细") == std::string::npos && 
                                  input.find("解释") == std::string::npos && input.find("列出") == std::string::npos;
                
                // 在LaunchResponseTokens之前设置上下文的实时性，确保在EageServe调度中正确应用
                int handleId = -1;
                
                try {
                    // 设置是否为实时任务
                    if (model->enableEageServe) {
                        std::cout << "启动任务 (" << (session->isRealtime ? "实时" : "非实时") 
                                  << (isShortTask ? "短" : "长") << "): " 
                                  << (input.length() > 30 ? input.substr(0, 30) + "..." : input) << std::endl;
                    }
                    
                    // 启动响应生成
                    handleId = model->LaunchResponseTokens(tokens, genConfig);
                    
                    // 打印任务ID
                    std::cout << "任务启动成功，handleId: " << handleId << std::endl;
                    
                    // 保存handleId以便可以中断
                    {
                        std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                        session->handleId = handleId;
                    }
                    
                    if (handleId < 0) {
                        std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                        session->output = "服务器处理请求时出错，请稍后重试。\n<eop>\n";
                        session->status = 2;
                        session->isProcessing = false;
                        session->resultCV.notify_all();
                        return;
                    }
                    
                    // 设置超时机制
                    auto startTime = std::chrono::steady_clock::now();
                    const auto maxGenerationTime = std::chrono::seconds(30); // 30秒生成超时
                    
                    std::vector<float> results;
                    std::string partialOutput = "";
                    int tokensGenerated = 0;
                    
                    while (true) {
                        // 检查是否超时
                        auto now = std::chrono::steady_clock::now();
                        if (std::chrono::duration_cast<std::chrono::seconds>(now - startTime) > maxGenerationTime) {
                            std::cerr << "生成超时，中断请求 handleId: " << handleId << std::endl;
                            model->AbortResponse(handleId);
                            break;
                        }
                        
                        try {
                            int result = model->FetchResponseTokens(handleId);
                            if (result == -1) {
                                std::cout << "任务 #" << handleId << " 生成完成，共生成 " << tokensGenerated << " 个token" << std::endl;
                                break;
                            } else {
                                tokensGenerated++;
                                results.clear();
                                results.push_back(result);
                                partialOutput = model->weight.tokenizer.Decode(fastllm::Data(fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                                
                                std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                                session->output += partialOutput;
                            }
                            
                            // 检查是否被中断
                            {
                                std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                                if (session->status == 2) {
                                    model->AbortResponse(handleId);
                                    break;
                                }
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Error fetching response: " << e.what() << std::endl;
                            break;
                        }
                    }
                    
                    // 更新会话状态
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                    if (!session->output.empty()) {
                        session->messages.push_back(std::make_pair("assistant", session->output));
                    } else {
                        session->output = "服务器生成回复时出错或超时。\n";
                    }
                    
                    session->output += "<eop>\n";
                    session->status = 2; // 设置为已完成
                    session->isProcessing = false;
                    session->handleId = -1; // 清除handleId
                    
                    // 通知等待的线程
                    session->resultCV.notify_all();
                    
                    // 打印生成完成日志
                    std::cout << "生成完成，会话 " << &session << " 已更新状态" << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Exception in chat thread: " << e.what() << std::endl;
                    
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                    session->output = "服务器处理请求时出错: " + std::string(e.what()) + "\n<eop>\n";
                    session->status = 2;
                    session->isProcessing = false;
                    session->resultCV.notify_all();
                    
                    // 如果已经启动了请求，尝试中止它
                    if (handleId >= 0) {
                        try {
                            model->AbortResponse(handleId);
                        } catch (...) {
                            // 忽略中止时的错误
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception in chat thread: " << e.what() << std::endl;
                
                std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                session->output = "服务器处理请求时出错，请稍后重试。\n<eop>\n";
                session->status = 2;
                session->isProcessing = false;
                session->resultCV.notify_all();
            }
        };

        svr.Post("/chat", [&](const httplib::Request &req, httplib::Response &res) {
            try {
                const std::string uuid = req.get_header_value("uuid");
                if (uuid.empty()) {
                    res.status = 400;
                    res.set_content("缺少uuid请求头", "text/plain");
                    return;
                }
                
                ChatSession* session = nullptr;
                
                // 获取或创建会话
                {
                    std::lock_guard<std::mutex> lock(sessionsLocker);
                    if (sessions.find(uuid) == sessions.end()) {
                        sessions[uuid] = new ChatSession();
                        sessions[uuid]->messages.push_back({"system", config.systemPrompt});
                        
                        // 根据请求头判断是否为实时任务
                        if (req.has_header("X-Realtime")) {
                            bool isRealtime = (req.get_header_value("X-Realtime") == "true");
                            sessions[uuid]->isRealtime = isRealtime;
                            std::cout << "创建新会话 " << uuid << " (" 
                                      << (isRealtime ? "实时" : "非实时") << ")" << std::endl;
                        } else {
                            std::cout << "创建新会话 " << uuid << " (默认非实时)" << std::endl;
                        }
                    }
                    session = sessions[uuid];
                }
                
                // 更新会话的最后访问时间
                session->lastAccessTime = std::chrono::steady_clock::now();
                
                // 如果请求仅仅是拉取已有结果
                if (req.body.empty() || req.has_param("init")) {
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                    if (session->status == 2) {
                        // 如果有结果，返回结果
                        res.set_content(session->output, "text/plain");
                        session->status = 0;
                        session->output = "";
                    } else {
                        // 否则返回空或正在处理中的状态
                        res.set_content(session->status == 1 ? "正在处理中" : "", "text/plain");
                    }
                    return;
                }
                
                // 非阻塞检查会话状态
                {
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex, std::try_to_lock);
                    if (!sessionLock.owns_lock()) {
                        // 如果无法立即获取锁，说明会话正被其他线程使用，返回忙状态
                        res.status = 503;
                        res.set_content("服务器正忙，请稍后重试", "text/plain");
                        return;
                    }
                    
                    // 检查会话状态
                    if (session->status != 0) {
                        if (session->status == 2) {
                            // 如果有结果，返回结果
                            res.set_content(session->output, "text/plain");
                            session->status = 0;
                            session->output = "";
                        } else {
                            // 否则返回处理中状态
                            res.set_content("正在处理中", "text/plain");
                        }
                        return;
                    }
                    
                    // 设置会话状态为处理中
                    session->output = "";
                    session->status = 1;
                }
                
                // 打印日志
                std::cout << "开始处理会话 " << uuid << " 的请求: " 
                          << (req.body.length() > 30 ? req.body.substr(0, 30) + "..." : req.body) << std::endl;
                
                // 启动处理线程
                std::thread chat_thread(chat, session, req.body);
                chat_thread.detach();
                
                // 非阻塞等待结果，支持快速响应
                {
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                    if (session->resultCV.wait_for(sessionLock, std::chrono::milliseconds(3000), 
                            [&]{ return session->status == 2; })) {
                        // 如果结果已经准备好，直接返回
                        res.set_content(session->output, "text/plain");
                        session->status = 0;
                        session->output = "";
                    } else {
                        // 否则返回处理中状态
                        res.set_content("正在处理中", "text/plain");
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception in /chat handler: " << e.what() << std::endl;
                res.status = 500;
                res.set_content("服务器内部错误", "text/plain");
            }
        });

        // 添加终止生成的接口
        svr.Post("/abort", [&](const httplib::Request &req, httplib::Response &res) {
            try {
                const std::string uuid = req.get_header_value("uuid");
                if (uuid.empty()) {
                    res.status = 400;
                    res.set_content("缺少uuid请求头", "text/plain");
                    return;
                }
                
                // 获取会话
                ChatSession* session = nullptr;
                {
                    std::lock_guard<std::mutex> lock(sessionsLocker);
                    auto it = sessions.find(uuid);
                    if (it == sessions.end()) {
                        res.status = 404;
                        res.set_content("会话不存在", "text/plain");
                        return;
                    }
                    session = it->second;
                }
                
                // 中断生成
                {
                    std::unique_lock<std::mutex> sessionLock(session->sessionMutex);
                    if (session->handleId >= 0) {
                        try {
                            model->AbortResponse(session->handleId);
                            res.set_content("已中断生成", "text/plain");
                        } catch (const std::exception& e) {
                            std::cerr << "中断生成时出错: " << e.what() << std::endl;
                            res.status = 500;
                            res.set_content("中断生成时出错", "text/plain");
                        }
                    } else {
                        res.set_content("无正在进行的生成任务", "text/plain");
                    }
                    
                    // 标记状态为已完成
                    session->status = 2;
                    session->isProcessing = false;
                    session->resultCV.notify_all();
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception in /abort handler: " << e.what() << std::endl;
                res.status = 500;
                res.set_content("服务器内部错误", "text/plain");
            }
        });
        
        // 设置请求超时
        svr.set_read_timeout(10); // 设置读取超时10秒
        svr.set_write_timeout(10); // 设置写入超时10秒

        svr.set_mount_point("/", config.webPath);
        std::cout << ">>> please open http://127.0.0.1:" + std::to_string(config.port) + "\n";
        std::cout << ">>> 服务已启动，支持实时和非实时请求处理\n";
        
        // 启动服务器
        svr.listen("0.0.0.0", config.port);
        
        // 服务器停止后执行清理工作
        std::cout << "正在关闭服务器..." << std::endl;
        
        // 停止清理线程
        cleanupThreadRunning = false;
        
        // 清理所有会话
        {
            std::lock_guard<std::mutex> lock(sessionsLocker);
            for (auto& pair : sessions) {
                try {
                    // 尝试中断所有正在处理的请求
                    std::unique_lock<std::mutex> sessionLock(pair.second->sessionMutex, std::try_to_lock);
                    if (sessionLock.owns_lock() && pair.second->handleId >= 0) {
                        model->AbortResponse(pair.second->handleId);
                    }
                    delete pair.second;
                } catch (...) {
                    // 忽略清理过程中的错误
                }
            }
            sessions.clear();
        }
        
        std::cout << "服务器已关闭" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
