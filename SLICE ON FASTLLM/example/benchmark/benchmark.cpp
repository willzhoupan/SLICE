//
// Created by huangyuyang on 6/9/23.
//

#include "model.h"
#include "utils.h"
#include "fstream"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <sstream>
#include <string>


#if defined(_WIN32) or defined(_WIN64)
#include <codecvt>

//GBK locale name in windows
const char* GBK_LOCALE_NAME = ".936";

std::string utf8_to_gbk(const std::string& str)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::wstring tmp_wstr;
    try {
        tmp_wstr = conv.from_bytes(str);
    } catch (const std::range_error& e) {
        return str;
    }
    std::wstring_convert<std::codecvt_byname<wchar_t, char, mbstate_t>> convert(new std::codecvt_byname<wchar_t, char, mbstate_t>(GBK_LOCALE_NAME));
    return convert.to_bytes(tmp_wstr);
}
#endif

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

struct BenchmarkConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    int threads = 4; // 使用的线程数
    int limit = -1; // 输出token数限制，如果 < 0 则代表无限制
    int batch = -1; // batch数, -1时使用文件中的行数作为batch
    std::string file; // 输入文件
    std::string output; // 输出文件，如果不设定则输出到屏幕
    bool enableSLICE = false; // 是否启用SLICE调度
    int longTaskThreshold = 128; 
    float priorityDegradeThreshold = 1.5f;
    bool testMode = false; // 测试模式
    int concurrentTasks = 4; // 并发任务数
    int totalTasks = 0; // 总任务数
    
    // 泊松分布参数
    float poissonLambda = 0.1f; // 泊松分布的lambda参数，默认为0.1
    bool usePoissonArrival = false; // 是否使用泊松分布的任务到达模式
    std::vector<float> predefinedIntervals; // 预定义的任务间隔时间
    int groupCnt = -1;
    double ratios = 1.0; // 超参系数

    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
};

// 生成指数分布的随机数，用于模拟泊松过程中的任务间隔时间
float generateExponentialInterval(float lambda) {
    // 使用C++11标准库中的随机数生成器
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::exponential_distribution<float> distribution(lambda);
    
    // 生成一个符合指数分布的随机数
    return distribution(gen);
}

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--limit> <args>:          输出token数限制" << std::endl;
    std::cout << "<-b|--batch> <args>:          batch数"      << std::endl;
    std::cout << "<-f|--file> <args>:           输入文件，文件中每行一个prompt，如果行数不足batch则用之前的prompt补充"      << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--slice>:                启用SLICE调度器" << std::endl;
    std::cout << "<--test-mode>:                启用SLICE测试模式" << std::endl;
    std::cout << "<--concurrent> <args>:        设置并发任务数(测试模式)" << std::endl;
    std::cout << "<--total-tasks> <args>:       设置总任务数(测试模式)" << std::endl;
    std::cout << "<--poisson>:                  启用泊松分布的任务到达模式" << std::endl;
    std::cout << "<--lambda> <args>:            设置泊松分布的lambda参数" << std::endl;
    std::cout << "<--intervals> <args>:         设置预定义的任务到达间隔时间(逗号分隔的浮点数)" << std::endl;
    std::cout << "<--ratios> <args>:            超参系数(浮点数)" << std::endl;
}   

void ParseArgs(int argc, char **argv, BenchmarkConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }

    // 预定义的泊松分布到达间隔时间（秒）
    std::map<double, std::vector<float>> arrivalIntervals = {

    };
    
    bool userSpecifiedIntervals = false;



    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        }
        else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--limit") {
            config.limit = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-b" || sargv[i] == "--batch") {
            config.batch = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-f" || sargv[i] == "--file") {
            config.file = sargv[++i];
        } else if (sargv[i] == "-o" || sargv[i] == "--output") {
            config.output = sargv[++i];
        } else if (sargv[i] == "--slice") {
            config.enableSLICE = true;
        } else if (sargv[i] == "--long-threshold") {
            config.longTaskThreshold = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--degrade-threshold") {
            config.priorityDegradeThreshold = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--test-mode") {
            config.testMode = true;
        } else if (sargv[i] == "--concurrent") {
            config.concurrentTasks = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--total-tasks") {
            config.totalTasks = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "--poisson") {
            config.usePoissonArrival = true;
        } else if (sargv[i] == "--lambda") {
            config.poissonLambda = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--intervals") {
            std::string intervalStr = sargv[++i];
            std::stringstream ss(intervalStr);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    config.predefinedIntervals.push_back(atof(item.c_str()));
                } catch (const std::exception& e) {
                    std::cerr << "解析间隔时间失败: " << item << " - " << e.what() << std::endl;
                }
            }
        } else if (sargv[i] == "--ratios") {
            config.ratios = atof(sargv[++i].c_str());   
        } else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "--atype") {
            std::string atypeStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
                                    "Unsupport act type: " + atypeStr);
            config.atype = dataTypeDict[atypeStr];
        } else {
            Usage();
            exit(-1);
        }
    }
    

    // 如果启用了泊松分布模式，但用户没有显式指定间隔时间，则使用预定义的间隔时间
    if (config.usePoissonArrival && !userSpecifiedIntervals && config.predefinedIntervals.empty()) {
        // 查找最接近用户指定lambda值的预定义间隔时间
        double minDiff = std::numeric_limits<double>::max();
        double closestLambda = 0.1; // 默认值
        
        for (const auto& pair : arrivalIntervals) {
            double diff = std::abs(pair.first - config.poissonLambda);
            if (diff < minDiff) {
                minDiff = diff;
                closestLambda = pair.first;
            }
        }
        
        // 填充预定义间隔时间
        config.predefinedIntervals = arrivalIntervals[closestLambda];
        std::cout << "使用预定义的泊松分布间隔时间 (lambda=" << closestLambda << ")" << std::endl;
    }



}

// 用于测试的任务结构
struct TestTask {
    std::string prompt;
    bool isRealtime;
    bool isShortTask;
    int priority; // 0-3: 实时短，实时长，非实时短，非实时长
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point allstartTime;
    std::chrono::steady_clock::time_point endTime;
    std::chrono::steady_clock::time_point allendTime;
    double interval;
    double latency; // 毫秒
    double alltime;
    int tokenCount;
    int handleId;
    bool completed;
    int reward;
    int TPOT; 
    std::string output = "";
    int typeIndex = -1; // 新增：任务类型索引
};

// 任务完成统计
struct TestStats {
    std::mutex statsMutex;
    double totalLatency[4] = {0, 0, 0, 0};
    int totalTasks[4] = {0, 0, 0, 0};
    int totalTokens[4] = {0, 0, 0, 0};

    
    
    void addResult(const TestTask& task) {
        std::lock_guard<std::mutex> lock(statsMutex);
        totalLatency[task.priority] += task.latency;
        totalTasks[task.priority]++;
        totalTokens[task.priority] += task.tokenCount;
    }

};

// 运行SLICE测试
void runSLICETest(fastllm::basellm* model, const BenchmarkConfig& config, bool forceDisableSLICE = false) {
    std::cout << "开始SLICE调度器测试...\n";
    std::cout << "========== 参数配置 ==========\n";
    std::cout << "并发任务数: " << config.concurrentTasks << "\n";
    std::cout << "超参系数: " << config.ratios << "\n";
    std::cout << "SLICE实际状态: " << (model->enableSLICE && !forceDisableSLICE ? "启用" : "禁用") << "\n";
    
    // 显示泊松分布相关配置
    if (config.usePoissonArrival) {
        std::cout << "任务到达模式: 泊松分布\n";
        std::cout << "Lambda参数: " << config.poissonLambda << "\n";
        if (!config.predefinedIntervals.empty()) {
            std::cout << "使用预定义的间隔时间序列，共 " << config.predefinedIntervals.size() << " 个间隔\n";
        }
    } else {
        std::cout << "任务到达模式: 同时到达\n";
    }
    
    std::cout << "============================\n\n";
    
    // 如果需要强制禁用SLICE，保存原始状态并暂时禁用
    bool originalSLICEState = model->enableSLICE;
    if (forceDisableSLICE && model->enableSLICE) {
        std::cout << "临时禁用SLICE进行测试" << std::endl;
        model->enableSLICE = false;
    }
    
    // 声明必要的变量
    std::mutex cvMutex;
    std::mutex taskStatusMutex;
    std::condition_variable cv;
    std::atomic<int> completedTasks(0);
    std::atomic<int> submittedTasks(0); // 跟踪已提交的任务数量
    std::unordered_map<int, int> handleToTaskIndex; // 处理ID到任务索引的映射
    std::unordered_set<int> activeHandles; // 当前活跃的任务handles
    TestStats stats; // 统计结果
    
    std::vector<TestTask> testTasks = {

        {
            "Describe your ideal vacation. Include the destination, one activity you would do, and why you chose it. Keep the description within 100 words.",
            false, false, 1, {}, {}, {}, {}, 0.04363281, 0, 0, 0, -1, false, 1, 120
        },
        {
            "Describe your ideal vacation. Include the destination, one activity you would do, and why you chose it. Keep the description within 100 words.",
            false, false, 1, {}, {}, {}, {}, 0.04363281, 0, 0, 0, -1, false, 1, 120
        },
        {
            "Describe your ideal vacation. Include the destination, one activity you would do, and why you chose it. Keep the description within 100 words.",
            false, false, 1, {}, {}, {}, {}, 0.04363281, 0, 0, 0, -1, false, 1, 120
        },
        {
            "Describe your ideal vacation. Include the destination, one activity you would do, and why you chose it. Keep the description within 100 words.",
            false, false, 1, {}, {}, {}, {}, 0.04363281, 0, 0, 0, -1, false, 1, 120
        },
        

    };
    
    // 准备测试任务队列
    std::vector<TestTask> tasks;

    std::cout << "实际测试任务数: " << (int)testTasks.size() << std::endl;

    // 顺序推入每个任务
    for (int i = 0; i < testTasks.size(); i++) {
        tasks.push_back(testTasks[i]);
    }
    
    
    
    // 输出任务分配情况
    int typeCounts[4] = {0, 0, 0, 0};
    for (const auto& task : tasks) {
        typeCounts[task.priority]++;
    }
    
    std::cout << "任务分配情况：" << std::endl;
    std::cout << "实时短任务 (优先级0): " << typeCounts[0] << " 个" << std::endl;
    std::cout << "实时长任务 (优先级1): " << typeCounts[1] << " 个" << std::endl;
    std::cout << "非实时短任务 (优先级2): " << typeCounts[2] << " 个" << std::endl;
    std::cout << "非实时长任务 (优先级3): " << typeCounts[3] << " 个" << std::endl;
    
    // 每个任务已生成的token数
    std::vector<int> tokensGenerated(tasks.size(), 0);
    
    // 为每个任务准备输入数据并加入队列
    std::cout << "准备所有任务并加入队列..." << std::endl;
    
    // 预处理所有任务的输入数据
    std::vector<std::vector<int>> allTokens;
    std::vector<fastllm::GenerationConfig> allGenConfigs;
    
    for (int i = 0; i < tasks.size(); i++) {
        TestTask& task = tasks[i];
        
        // 编码输入
        fastllm::ChatMessages messages;
        messages.push_back({"user", task.prompt});
        std::string prompt;

        std::cout << prompt << std::endl;
        
        try {
            prompt = model->ApplyChatTemplate(messages);
        } catch (const std::exception& e) {
            std::cerr << "任务 #" << i << " 应用聊天模板失败: " << e.what() << std::endl;
            prompt = task.prompt; // 使用原始提示作为后备
        }
        
        auto inputData = model->weight.tokenizer.Encode(prompt);
        if (inputData.dims.empty() || inputData.Count(0) == 0) {
            std::cerr << "任务 #" << i << " 编码输入失败，跳过该任务" << std::endl;
            // 将空向量添加为占位符
            allTokens.push_back(std::vector<int>());
            allGenConfigs.push_back(fastllm::GenerationConfig());
            continue;
        }
        
        // 准备输入tokens
        std::vector<int> tokens;
        for (int j = 0; j < inputData.Count(0); j++) {
            tokens.push_back(((float*)inputData.cpuData)[j]);
        }
        
        // 设置生成配置
        fastllm::GenerationConfig genConfig;
        
        // 根据任务优先级设置不同的token输出限制
        switch (task.priority) {
            case 0: // 实时短任务
                genConfig.output_token_limit = -1;
                break;
            case 1: // 实时长任务
                genConfig.output_token_limit = -1;
                break;
            case 2: // 非实时短任务
                genConfig.output_token_limit = -1;
                break;
            case 3: // 非实时长任务
                genConfig.output_token_limit = -1;
                break;
            default:
                genConfig.output_token_limit = -1; // 默认值
        }
        
        // 根据任务类型设置优先级属性
        genConfig.priority = task.priority;
        genConfig.isRealtime = task.isRealtime;
        genConfig.isShortTask = task.isShortTask;
        genConfig.reward = task.reward;
        genConfig.TPOT = task.TPOT;
        
        // 存储预处理好的数据
        allTokens.push_back(tokens);
        allGenConfigs.push_back(genConfig);
    }
    
    // 1. 延迟调度
    //model->DelayScheduling(true); // 暂停调度

    // 初始化outputs向量，确保有足够的空间
    std::vector<std::string> outputs(tasks.size());
    int index = 0;

    // 生成或使用预定义的任务间隔时间
    std::vector<float> intervalTimes;
    if (config.usePoissonArrival) {
        if (!config.predefinedIntervals.empty()) {
            // 使用预定义的间隔时间
            intervalTimes.resize(tasks.size());
            for (size_t i = 0; i < tasks.size(); ++i) {
                int idx = tasks[i].typeIndex >= 0 ? tasks[i].typeIndex : 0;
                intervalTimes[i] = config.predefinedIntervals[idx];
            }
            std::cout << "使用预定义的任务间隔时间序列(按类型):" << std::endl;
            for (size_t i = 0; i < 10; i++) {
                std::cout << config.predefinedIntervals[i] << " ";
            }
            std::cout << std::endl;
        } else {
            // 生成符合指数分布的间隔时间
            std::cout << "生成符合指数分布的任务间隔时间 (lambda=" << config.poissonLambda << "):" << std::endl;
            for (size_t i = 0; i < tasks.size() - 1; i++) {
                float interval = generateExponentialInterval(config.poissonLambda);
                intervalTimes.push_back(interval);
                if (i < 10) {
                    std::cout << std::fixed << std::setprecision(4) << interval << " ";
                }
            }
            if (tasks.size() > 10) {
                std::cout << "...";
            }
            std::cout << std::endl;
        }
    }

    // 启动调度器
    std::cout << "启动调度器，任务将在提交后立即开始处理..." << std::endl;
    model->StartScheduler();


    // 根据泊松分布或同时到达模式提交任务
    if (config.usePoissonArrival) {

        auto totaltasktime =  std::chrono::system_clock::now();

        std::cout << "创建任务提交线程，按类型间隔提交任务..." << std::endl;
        std::thread submitThread([&]() {
        for (int i = 0; i < tasks.size(); i++) {
            TestTask& task = tasks[i];
            
            // 如果预处理阶段失败，跳过该任务
            if (allTokens[i].empty()) {
                // 标记任务完成
                {
                    std::lock_guard<std::mutex> lock(cvMutex);
                    completedTasks++;
                }
                cv.notify_one();
                continue;
            }
            
            // 记录任务开始时间
            task.allstartTime = std::chrono::steady_clock::now();
            task.startTime = std::chrono::steady_clock::now();
            
            // 启动任务并加入调度队列
            try {
                std::cout << "启动任务 #" << i << " (" 
                          << (task.isRealtime ? "实时" : "非实时") << (task.isShortTask ? "短" : "长") 
                          << "任务) 优先级: " << task.priority << std::endl;
                          // add by zp
                           auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(task.startTime.time_since_epoch()).count();
                           std::cout << "starttime = " << start_ms << " ms" << std::endl;
                
                task.handleId = model->LaunchResponseTokens(allTokens[i], allGenConfigs[i], config.ratios , config.batch);
                
                if (task.handleId >= 0) {
                    // 记录任务handle和索引的映射关系
                    std::lock_guard<std::mutex> lock(taskStatusMutex);
                    handleToTaskIndex[task.handleId] = i;
                    activeHandles.insert(task.handleId);
                    submittedTasks++; // 增加已提交任务计数
                    std::cout << "任务 #" << i << " 启动成功，handleId: " << task.handleId << std::endl;
                } else {
                    std::cerr << "任务 #" << i << " 启动失败，handleId: " << task.handleId << std::endl;
                    // 标记任务完成
                    {
                        std::lock_guard<std::mutex> lock(cvMutex);
                        completedTasks++;
                    }
                    cv.notify_one();
                }
            } catch (const std::exception& e) {
                std::cerr << "任务 #" << i << " LaunchResponseTokens抛出异常: " << e.what() << std::endl;
                // 标记任务完成
                {
                    std::lock_guard<std::mutex> lock(cvMutex);
                    completedTasks++;
                }
                cv.notify_one();
            }

            printf("Task %d added (Tokens: %zu, Priority: %d)\n", 
           task.handleId, allTokens[i].size(), allGenConfigs[i].priority);
            
            // 如果不是最后一个任务，等待指定的间隔时间
            if (i < tasks.size() - 1) {
                float waitTime = task.interval;
                std::cout << "等待 " << std::fixed << std::setprecision(4) << waitTime << " 秒后提交下一个任务..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(waitTime * 1000)));
            }
        }
        });
        
        // 分离线程，让它在后台运行
        submitThread.detach();
        std::cout << "任务提交线程已启动，继续监控任务执行..." << std::endl;
    } else {
        // 同时到达模式：一次性提交所有任务
        submittedTasks = 0; // 初始化已提交任务计数
        for (int i = 0; i < tasks.size(); i++) {
            TestTask& task = tasks[i];
            
            // 如果预处理阶段失败，跳过该任务
            if (allTokens[i].empty()) {
                // 标记任务完成
                {
                    std::lock_guard<std::mutex> lock(cvMutex);
                    completedTasks++;
                }
                cv.notify_one();
                continue;
            }
            
            // 记录任务开始时间
            task.allstartTime = std::chrono::steady_clock::now();
            task.startTime = std::chrono::steady_clock::now();
            
            // 启动任务并加入调度队列
            try {
                std::cout << "启动任务 #" << i << " (" 
                          << (task.isRealtime ? "实时" : "非实时") << (task.isShortTask ? "短" : "长") 
                          << "任务) 优先级: " << task.priority << std::endl;
                          
                std::ofstream fout("任务启动和结束.txt", std::ios::app);
                
                fout << "启动任务 #" << i << " (" 
                          << (task.isRealtime ? "实时" : "非实时") << (task.isShortTask ? "短" : "长") 
                          << "任务) 优先级: " << task.priority << std::endl;
                fout.close();
                
                task.handleId = model->LaunchResponseTokens(allTokens[i], allGenConfigs[i], config.ratios , config.batch);
                
                if (task.handleId >= 0) {
                    // 记录任务handle和索引的映射关系
                    std::lock_guard<std::mutex> lock(taskStatusMutex);
                    handleToTaskIndex[task.handleId] = i;
                    activeHandles.insert(task.handleId);
                    submittedTasks++; // 增加已提交任务计数
                    std::cout << "任务 #" << i << " 启动成功，handleId: " << task.handleId << std::endl;
                } else {
                    std::cerr << "任务 #" << i << " 启动失败，handleId: " << task.handleId << std::endl;
                    // 标记任务完成
                    {
                        std::lock_guard<std::mutex> lock(cvMutex);
                        completedTasks++;
                    }
                    cv.notify_one();
                }
            } catch (const std::exception& e) {
                std::cerr << "任务 #" << i << " LaunchResponseTokens抛出异常: " << e.what() << std::endl;
                // 标记任务完成
                {
                    std::lock_guard<std::mutex> lock(cvMutex);
                    completedTasks++;
                }
                cv.notify_one();
            }

            printf("Task %d added (Tokens: %zu, Priority: %d)\n", 
           task.handleId, allTokens[i].size(), allGenConfigs[i].priority);
        }
    }


    
    // 超时控制
    auto startTime = std::chrono::steady_clock::now();
    const auto maxProcessTime = std::chrono::seconds(6000); // 最长处理10分钟

    
    
    // 处理循环 - 直到所有任务完成或超时
    while (completedTasks < submittedTasks || (config.usePoissonArrival && submittedTasks < tasks.size())) {
        // 检查是否超时
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - startTime) > maxProcessTime) {
            std::cerr << "任务处理超时，已完成 " << completedTasks << "/" << submittedTasks << " 个任务" << std::endl;
            break;
        }


        // 复制当前活跃的handles列表，避免锁竞争
        std::vector<int> currentHandles;
        {
            std::lock_guard<std::mutex> lock(taskStatusMutex);
            currentHandles.assign(activeHandles.begin(), activeHandles.end());
        }

        // //打印当前活跃的handles列表
        // std::cout << "当前活跃的currentHandles列表: ";
        // for (int handle : currentHandles) {
        //     std::cout << handle << " ";
        // }
        // std::cout << std::endl;
        
        if (currentHandles.empty()) {
            // 没有活跃任务，等待一段时间再检查
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // 处理当前活跃的任务
        for (int handle : currentHandles) {
            int taskIndex;
            {
                std::lock_guard<std::mutex> lock(taskStatusMutex);
                if (handleToTaskIndex.find(handle) == handleToTaskIndex.end()) {
                    continue; // 任务可能已被移除
                }
                taskIndex = handleToTaskIndex[handle];
                
                // 检查taskIndex是否有效
                if (taskIndex < 0 || taskIndex >= tasks.size()) {
                    std::cerr << "警告：无效的任务索引 " << taskIndex << "，跳过处理" << std::endl;
                    continue;
                }


            }
            
            try {
                // 获取生成的token
                int result = model->FetchResponseTokens(handle);
                std::vector<float> results;
                if (result == -1) {
                    // 任务完成
                    {
                        std::lock_guard<std::mutex> lock(taskStatusMutex);
                        // 再次检查索引有效性
                        if (taskIndex >= 0 && taskIndex < tasks.size() && taskIndex < tokensGenerated.size()) {
                            TestTask& task = tasks[taskIndex];

                            // 记录结束时间和统计数据
                            task.endTime = std::chrono::steady_clock::now();
                            auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(task.startTime.time_since_epoch()).count();
                            auto end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(task.endTime.time_since_epoch()).count();
                            auto dur = end_ms - start_ms;
                            task.latency = std::chrono::duration<double, std::milli>(task.endTime - task.startTime).count();
                            task.tokenCount = tokensGenerated[taskIndex];
                            task.allendTime = std::chrono::steady_clock::now();
                            task.alltime = std::chrono::duration<double, std::milli>(task.allendTime - task.allstartTime).count();
                            task.completed = true;

                            // 从活跃任务列表中移除
                            activeHandles.erase(handle);

                            // 添加到统计
                            stats.addResult(task);
                            task.output += "<eop>\n";

                            
                            // 确保index在outputs向量范围内
                            if (index < outputs.size()) {
                                outputs[index] += task.output;
                                index++;
                            } else {
                                std::cerr << "错误：outputs索引越界 (index=" << index << ", size=" << outputs.size() << ")" << std::endl;
                            }
                        } else {
                            std::cerr << "错误：任务索引越界 (taskIndex=" << taskIndex << ", tasks.size=" << tasks.size() 
                                    << ", tokensGenerated.size=" << tokensGenerated.size() << ")" << std::endl;
                            // 从活跃任务列表中移除无效的handle
                            activeHandles.erase(handle);
                        }

                        completedTasks++;
                    }

                    cv.notify_one();
                } else {
                    // 检查索引有效性
                    std::lock_guard<std::mutex> lock(taskStatusMutex);
                    if (taskIndex >= 0 && taskIndex < tasks.size() && taskIndex < tokensGenerated.size()) {
                        TestTask& task = tasks[taskIndex];
                        results.clear();
                        results.push_back(result);
                        task.output += model->weight.tokenizer.Decode(fastllm::Data(fastllm::DataType::FLOAT32, {(int)results.size()}, results));
                        // 任务仍在进行中，记录生成的token
                        tokensGenerated[taskIndex]++;
                    } else {
                        std::cerr << "错误：更新token时任务索引越界 (taskIndex=" << taskIndex << ", tasks.size=" << tasks.size() 
                                << ", tokensGenerated.size=" << tokensGenerated.size() << ")" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "处理任务 #" << taskIndex << " 时出错: " << e.what() << std::endl;
                
                // 标记任务异常结束
                {
                    std::lock_guard<std::mutex> lock(taskStatusMutex);
                    // 检查索引有效性
                    if (taskIndex >= 0 && taskIndex < tasks.size() && taskIndex < tokensGenerated.size()) {
                        TestTask& task = tasks[taskIndex];
                        task.endTime = std::chrono::steady_clock::now();
                        task.latency = std::chrono::duration<double, std::milli>(task.endTime - task.startTime).count();
                        task.tokenCount = tokensGenerated[taskIndex];
                        task.completed = true;
                        activeHandles.erase(handle);
                    } else {
                        std::cerr << "错误：处理异常时任务索引越界 (taskIndex=" << taskIndex << ")" << std::endl;
                        // 从活跃任务列表中移除无效的handle
                        activeHandles.erase(handle);
                    }
                }
                
                // 通知主线程任务完成
                {
                    std::lock_guard<std::mutex> lock(cvMutex);
                    completedTasks++;
                }
                cv.notify_one();
            }
        }
        
        // 短暂休眠，避免过度CPU占用
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    
    // 确保所有任务资源已清理
    try {
        // 安全清理SLICE调度器的状态
        if (model->enableSLICE) {
            model->scheduler.SafeClear();
        }
    } catch (const std::exception& e) {
        std::cerr << "清理SLICE调度器状态时出错: " << e.what() << std::endl;
    }
    
    // 恢复SLICE状态
    if (forceDisableSLICE && originalSLICEState) {
        model->enableSLICE = originalSLICEState;
    }

}

int main(int argc, char **argv) {
    
    BenchmarkConfig config;
    ParseArgs(argc, argv, config);
    fastllm::SetThreads(config.threads);

     if (!fastllm::FileExists(config.path)) {
        printf("模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    try {
        auto model = !isHFDir ? fastllm::CreateLLMModelFromFile(config.path) : fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt);
        if (config.atype != fastllm::DataType::FLOAT32) {
            model->SetDataType(config.atype);
        }
        
        // 配置SLICE
        if (config.enableSLICE) {
            model->enableSLICE = true;
        }

        // 如果启用测试模式，运行SLICE测试
        if (config.testMode) {
            if (!config.enableSLICE) {
                printf("警告：启用测试模式但未启用SLICE调度器。建议添加--slice参数。\n");
            }
            
            // 显示泊松分布配置信息
            if (config.usePoissonArrival) {
                printf("任务到达模式: 泊松分布 (lambda=%.4f)\n", config.poissonLambda);
                if (!config.predefinedIntervals.empty()) {
                    printf("使用预定义的间隔时间序列，共 %zu 个间隔\n", config.predefinedIntervals.size());
                    printf("前10个间隔时间: ");
                    for (size_t i = 0; i < config.predefinedIntervals.size() && i < 10; i++) {
                        printf("%.4f ", config.predefinedIntervals[i]);
                    }
                    if (config.predefinedIntervals.size() > 10) {
                        printf("...");
                    }
                    printf("\n");
                }
            } else {
                printf("任务到达模式: 同时到达\n");
            }
            
            try {

                
                // 尝试两种模式：先禁用SLICE运行一次，然后如果启用了再运行一次
                bool hasSLICE = model->enableSLICE;
                
                // 先强制禁用SLICE运行
                std::cout << "\n===== 禁用SLICE模式运行测试 =====\n" << std::endl;
                //runSLICETest(model.get(), config, true);

                // 如果启用了SLICE，再以启用状态运行一次
                if (hasSLICE) {
                    std::cout << "\n===== 启用SLICE模式运行测试 =====\n" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(2)); // 短暂等待，确保前一次测试资源释放
                    runSLICETest(model.get(), config, false);

                }
            } catch (const std::exception& e) {
                std::cerr << "SLICE测试失败: " << e.what() << std::endl;
            }
            return 0;
        }


        fastllm::GenerationConfig generationConfig;
        generationConfig.output_token_limit = config.limit;

        fastllm::PrintInstructionInfo();
        std::vector <std::string> inputs;
        if (config.file != "") {
            std::ifstream finputs(config.file, std::ios::in);
            if (finputs.good()) {
                while (true) {
                    std::string input = "";
                    std::getline(finputs, input);
                    if (input == "") {
                        break;
                    } else {
                        inputs.push_back(input);
                    }
                }
                finputs.close();
            }
        }
        if (inputs.empty()) {
            inputs.push_back("Hello!");
        }
        if (config.batch <= 0) {
            config.batch = inputs.size();
        }
        while (inputs.size() < config.batch) {
            inputs.push_back(inputs[rand() % inputs.size()]);
        }
        if (inputs.size() > config.batch && config.batch > 0) {
            inputs.resize(config.batch);
        }

        int promptTokenNum = 0;
        for (int i = 0; i < inputs.size(); i++) {
            fastllm::ChatMessages messages;
            messages.push_back({"user", inputs[i]});
            inputs[i] = model->ApplyChatTemplate(messages);
            promptTokenNum += model->weight.tokenizer.Encode(inputs[i]).Count(0);
        }

        std::vector <std::string> outputs;
        static int tokens = 0;
        auto st = std::chrono::system_clock::now();
        static auto promptTime = st;
        model->ResponseBatch(inputs, outputs, [](int index, std::vector<std::string> &contents) {
            if (index != -1) {
                if (index == 0) {
                    promptTime = std::chrono::system_clock::now();
                } else {
                    for (int i = 0; i < contents.size(); i++) {
                        tokens += (contents[i].size() > 0);
                    }
                }
            }
        }, generationConfig);
        float promptSpend = fastllm::GetSpan(st, promptTime);
        float spend = fastllm::GetSpan(promptTime, std::chrono::system_clock::now());

        if (config.output != "") {
            FILE *fo = fopen(config.output.c_str(), "w");
            for (int i = 0; i < outputs.size(); i++) {
                fprintf(fo, "[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
            }
            fclose(fo);
        } else {
            for (int i = 0; i < outputs.size(); i++) {
#if defined(_WIN32) or defined(_WIN64)
                printf("[ user: \"%s\", model: \"%s\"]\n", utf8_to_gbk(inputs[i]).c_str(), utf8_to_gbk(outputs[i]).c_str());
#else
                printf("[ user: \"%s\", model: \"%s\"]\n", inputs[i].c_str(), outputs[i].c_str());
#endif
            }
        }



        printf("batch: %d\n", (int)inputs.size());
        printf("prompt token number = %d\n", promptTokenNum);
        printf("prompt use %f s\n", promptSpend);
        printf("prompt speed = %f tokens / s\n", (float)promptTokenNum / promptSpend);
        printf("output %d tokens\nuse %f s\nspeed = %f tokens / s\n", tokens, spend, tokens / spend);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "程序运行出错: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "程序运行出现未知错误" << std::endl;
        return 1;
    }
}

