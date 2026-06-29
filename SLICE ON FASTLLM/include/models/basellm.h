#ifndef FASTLLM_BASELLM_H
#define FASTLLM_BASELLM_H

#include "fastllm.h"
#include "template.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <deque>
#include <atomic>
#include <array>

#ifdef PY_API
#include "Python.h"
#include <pybind11/pytypes.h>
using RuntimeResult = std::function<void(int index, pybind11::bytes content)>;
using RuntimeResultBatch = std::function<void(int index, std::vector <pybind11::bytes> &contents)>;
#else
using RuntimeResult = std::function<void(int index, const char* content)>;
using RuntimeResultBatch = std::function<void(int index, std::vector <std::string> &contents)>;
#endif

namespace fastllm {
    using ChatMessages = std::vector <std::pair <std::string, std::string> >;

    enum ResponseContextError {
        ResponseContextErrorNone = 0, ResponseContextErrorPromptTooLong
    };

    struct ResponseContext {
        bool isEnding = false; // 代表这个请求已经处理完成了，不需要再forward了，但生成的token可能还没有被fetch
        bool isAbort = false; // 代表这个请求被中断了，也就是说不会再有人来fetch它了，那么推理完之后就可以删除这个请求了
        
        std::vector <std::pair <Data, Data> > pastKeyValues;
        std::vector <int> currentTokens;
        std::queue <int> resultTokenQueue;
        std::queue <std::vector <float>*> resultLogits;
        GenerationConfig generationConfig;
        LastTokensUnit tokens;
        ResponseContextError error = ResponseContextErrorNone;

        std::chrono::system_clock::time_point notEGstartTime;

        int preTokens = 0;
        int curTokens = 0;
        std::map <std::string, int> intParams;

        int cacheLen = 0;

        // Prefill时间
        float prefillTime = 0.0f;

        std::chrono::system_clock::time_point ttf_start_time;
        std::chrono::system_clock::time_point slicetime;
        float ttf_time = 0.0f; // TTFT（秒）
        bool ttf_recorded = false; // 防止重复记录
        int reward = 0;
        int TPOT = 0;
        bool satisfySLO = false;
        bool isrecord = true;
        

        // EageServe相关属性
        bool isRealtime = false;    // 是否为实时任务
        bool isShortTask = true;    // 是否为短任务
        int priority = 2;           // 任务优先级(0-3)
        int tokenGenerated = 0;     // 已生成的token数量
        bool isPreempted = false;   // 是否被抢占

        void Init(int blocks, DataType dataType);
    };

    struct ResponseContextDict {
        std::mutex locker;
        std::map <int, ResponseContext*> dicts;

        int CreateHandle();

        ResponseContext* GetHandle(int handleId);

        void RemoveHandle(int handleId);
    };

    struct PastKVCacheMemory {
        std::vector <int> inputToken;
        int tokens;
        int recordTimes = 0;
        long long flushTime;
        std::vector<std::pair<Data, Data> > kv;

        PastKVCacheMemory () {}

        PastKVCacheMemory (const std::vector <int> &prompt, int tokens, long long flushTime, std::vector<std::pair<Data, Data> > *kv);
    };

    struct PastKVCacheManager {
        std::mutex locker;
        int maxRecordNum = 5;
        long long flushTime = 0;
        std::map <std::vector <int>, PastKVCacheMemory*> memorys;

        // 设置最多保存的记录条数
        void SetMaxRecordNum(int maxRecordNum);

        // 插入一条记录，若已存在则增加引用计数
        void Record(const std::vector <int> &inputToken, int tokens, std::vector<std::pair<Data, Data> > *kv);

        // 尝试删除一条记录，若引用计数非0不会真的删除
        void Remove(const std::vector <int> &inputToken);

        // 获取最长匹配的Memory，并加锁
        PastKVCacheMemory *Get(const std::vector <int> &inputToken);

        // 解锁
        void Unlock();
    };

    // 前向声明
    class EageServeScheduler;

    // SLICE类
    class SLICE {
    public:
        SLICE(EageServeScheduler* scheduler);
        ~SLICE();
        
        // 空函数1
        void SelectJob();
        
        // 空函数2
        void ConstructMatrix();

        // 空函数3
        void SelectMatrixJob();

        void RemoveMatrixJob(int handleId);

        void FindJob(double ratios);


        int dynamicMatrixSize = 0;  // 矩阵行数

        int dynamicMatrixIndex = 0;  // 矩阵列数

        bool Callmatrix = false;  //激活矩阵

        std::vector<std::array<int, 50>> dynamicMatrix;

        std::vector<std::pair<int, int>> taskList;  // {handleId, 1000/TPOT降序排序}

        int MatrixTokenMax = 0;

        int selecttotalToken = 0;



        
    private:

        std::mutex SLICEMutex;  // 互斥锁

        EageServeScheduler* scheduler;  // 添加 scheduler 指针

    
    };

    // EageServe调度器类
    class EageServeScheduler {
    public:
        EageServeScheduler();
        
        // 添加任务到调度队列
        void AddTask(int handleId, ResponseContext* context, bool isRealtime, bool isShortTask, int priority, int reward, int TPOT, int u);
        
        // 获取下一批要处理的任务
        int GetNextBatch(int tokencountindex, double ratios, double starvation_threshold, bool enableStarvation);
        
        // 处理任务完成
        void TaskCompleted(int handleId);
        
        // 处理任务被抢占
        void TaskPreempted(int handleId);
        
        // 更新任务优先级
        void UpdateTaskPriority(int handleId, int newPriority);
        
        // 预测平均输出长度
        int PredictAverageLength();
        
        // 计算最大批处理大小
        int CalculateMaxBatchSize(long long availableMemory, int tokenMemoryUsage);
        
        // 动态降级任务优先级
        void DegradePriority(ResponseContext* context,int handleId, int avgLength);

        // 新增：记录已完成任务的输出token数
        void RecordTaskLength(int length);

        void LogSchedulerStatus(int job_num,int handleId,const std::string& filename = "scheduler_status.txt");

        void LogSchedulerStatusPrint(int job_num,int handleId);

         // 添加安全清理方法
        void SafeClear() {
        for (int i = 0; i < 4; i++) {
            priorityQueues[i].clear();
        }
         handleToPriority.clear();
         currentBatch.clear();
         historicalLengths.clear();

         } 
    

        // 当前批处理中的任务
        std::set<int> currentBatch;

        // 四个优先级队列
        std::deque<int> priorityQueues[4];

        SLICE slice; 




        

        long long last_latency_getnextbatch_ms = -1;
        long long last_latency_forward_ms = -1;

     // 任务到开始时间的映射
        std::map<int, std::chrono::system_clock::time_point> handleStartTime;
        
        std::map<int, ResponseContext*> handleToContext;

        struct TaskStats {
        std::mutex statsMutex;
        double totalLatency[4] = {0, 0, 0, 0};
        int totalTasks[4] = {0, 0, 0, 0};
        int totalTokens[4] = {0, 0, 0, 0};

        void addResult(int TPOT, double latency, int tokens) {
            std::lock_guard<std::mutex> lock(statsMutex);
            if(TPOT == 50) {
            totalLatency[1] += latency;
                totalTasks[1]++;
                totalTokens[1] += tokens;   
            }
            else if(TPOT == 100) {
                totalLatency[2] += latency;
                totalTasks[2]++;
                totalTokens[2] += tokens;
            }
            else if(TPOT == 250) {
                totalLatency[3] += latency;
                totalTasks[3]++;
                totalTokens[3] += tokens;
            }
        }

        void printStats() {
            std::lock_guard<std::mutex> lock(statsMutex);
            std::cout << "\nadd by zp EG================== EageServe 测试结果统计 ==================\n";
            std::cout << std::setw(15) << "任务类型" << std::setw(12) << "任务数" << std::setw(15) << "平均时间(毫秒)" << std::setw(12) <<"总时间(毫秒)" << std::setw(12) << "总Token数\n";
            const char* taskTypes[] = {"实时短任务", "实时长任务", "非实时短任务", "非实时长任务"};
            for (int i = 0; i < 4; i++) {
                double avgLatency = totalTasks[i] > 0 ? totalLatency[i] / totalTasks[i] : 0;
                double totalTime = totalLatency[i];
                std::cout << std::setw(15) << taskTypes[i] << std::setw(12) << totalTasks[i] 
                        << std::setw(15) << std::fixed << std::setprecision(2) << avgLatency
                        << std::setw(12) << totalTime << std::setw(12) << totalTokens[i] << "\n";
            }
            std::cout << "==================================================\n";
        }
        };

        TaskStats EGstats;

        // 当前批处理中的任务优先级的映射
        std::vector<std::pair<int, int> > currentBatchPriority;

    



        std::unordered_map<std::string, std::pair<float, float>> task_type_thresholds;

        std::chrono::system_clock::time_point StarvationTime_A;

        std::chrono::system_clock::time_point StarvationTime_B;

        int totalTasks = 70; // 总任务数
        std::atomic<int> completedTasks{0}; // 已完成任务数，线程安全
        
    private:

        
        // 任务句柄到优先级的映射
        std::map<int, int> handleToPriority;
        
        // 历史任务输出长度记录
        std::deque<int> historicalLengths;
        
        
        // 互斥锁
        std::mutex schedulerMutex;



        // 当前批处理中的最低优先级
        int currentBatchHighestPriority = 0;
        // 平均长度预测的LMS参数
        float lmsAlpha = 0.2f;
        float predictedAvgLength = 128.0f;
    };

    enum RoPEType { // 位置编码外推类型
        BASE = 0,
        LINEAR_SCALE = 1,
        STATIC_NTK = 2,
        DYMAMIC_NTK = 3,
        YARN = 4
    };

    class basellm {
    public:
        // 新增调度控制接口
        void DelayScheduling(bool delay) { delayScheduling = delay; }
        void StartScheduler(); // 显式启动调度

        basellm() {};

        ~basellm();

        virtual void LoadFromFile(const std::string &fileName); // 从文件读取 

        virtual void InitParams(); // 初始化参数信息 

        // 根据原始的tensorNames获得映射表
        virtual std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(const std::vector <std::string> &tensorNames);

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr) = 0;

        virtual std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        virtual std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr,
                bool enableEageServe = false,
                int tokencount = 0);

        // 是否需要生成AttentionMask
        virtual bool NeedAttentionMask(int qlen, int klen);

        // 根据输入的tokens生成LLM推理的输入 
        virtual void FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                   const std::map <std::string, int> &params,
                                   Data &inputIds, Data &attentionMask, Data &positionIds);

        // 根据输入的tokens生成LLM推理的输入 
        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual std::string Response(const std::string &input,
                                     RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig());

        virtual void ResponseBatch(const std::vector<std::string> &inputs,
                                   std::vector<std::string> &outputs,
                                   RuntimeResultBatch retCb = nullptr,
                                   const GenerationConfig &generationConfig = GenerationConfig()); // 批量根据给出的内容回复 

        virtual int LaunchResponseTokens(const std::vector<int> &inputTokens,
                                         const GenerationConfig &generationConfig = GenerationConfig(), // 启动一个response任务，返回分配的handleId
                                         double ratios = 1.0,
                                         double starvation_threshold = 0.0,
                                         int batch = 1,
                                         bool enableStarvation = false);

        virtual bool CanFetchResponse(int handleId); // 判断当前是否能fetch到，用于异步操作

        virtual int FetchResponseTokens(int handleId); // 获取指定handle的输出, -1代表输出结束了 

        virtual int FetchResponseLogits(int handleId, std::vector <float> &logits); // 获取指定handle的输出Logits

        virtual void AbortResponse(int handleId); // 中断handleId的请求

        virtual void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型 

        virtual void SaveModel(const std::string &fileName); // 直接导出

        virtual void WarmUp() {}; // 预热

        virtual void AddPromptCache(const std::vector <int> &inputTokens);

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input) = 0; // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) = 0; // 根据当前回复更新history

        virtual void SetAdapter(const std::string &name);

        virtual void DisableAdapter();

        virtual bool SetSaveHistoryChat(bool save);

        virtual void SetDataType(DataType dataType);

        // messages: [ (role, content) ... ]
        virtual std::string ApplyChatTemplate(const ChatMessages &messages);

        virtual std::vector <int> ApplyChatTemplateToTokens(const ChatMessages &messages);

        virtual std::string ApplyChatTemplate(const JinjaVar &var);

        virtual std::vector <int> ApplyChatTemplateToTokens(const JinjaVar &var);

        std::string model_type;
        std::string model_struct;

        std::string pre_prompt; // 最初对话的提示语
        std::string user_role, bot_role, history_sep; // 用于生成每一轮的prompt

        int bos_token_id;
        int eos_token_id;
        std::set <int> eos_token_ids;
        int embed_dim = 4096;
        int num_attention_heads = 32;
        int head_dim = embed_dim / num_attention_heads;
        int max_positions = 32768;
        int rotary_dim = 64;
        const float scale_attn = sqrt(head_dim);
        int block_cnt = 28;

        std::vector<std::vector<float> > sin, cos;

        WeightMap weight; // 权重

        Data sinData, cosData;

        ResponseContextDict responseContextDict;

        std::thread *mainLoop = nullptr;
        std::mutex mainLoopLocker, dictLocker;
        std::condition_variable dictCV;

        std::map <std::string, int> deviceMap;

        std::string adapterName;

        int tokensLimit = -1;
        int promptLimit = -1;

        PastKVCacheManager pastKVCacheManager;
        bool saveHistoryChat = false;

        std::string lastPrompt = "";
        std::vector<std::pair<Data, Data> > *lastKeyValues = nullptr;
        int lastPromptTokens = 0;
        
        long long elementsInKVCachePerToken = -1; // 每个token使用多少个元素的的KVCache
        long long kvCacheLimit = -1;
        int maxBatch = 1;
        bool verbose = true;
        

        DataType dataType = DataType::FLOAT32;
        bool isFree = false; // 是否释放
        
        // EageServe调度器
        EageServeScheduler scheduler;
        
        // 是否启用EageServe调度
        bool enableEageServe = false;
        
        // 单个Token的显存开销估计(字节)
        long long tokenMemoryUsage = 917504; 

        long long modelweightMemoryUsage = 3100000000/1024/1024;
        
        // 任务长度阈值，超过此值的任务被视为长任务
        int longTaskThreshold = 128;
        
        // 优先级降级阈值，超过平均长度的倍数
        float priorityDegradeThreshold = 1.5f;
        
        int EG_all_Throughput = 0;

        int EG_Throughput_avg = 0;

        int EG_Throughput_count = 0;

        int EG_all_forward_count = 0;

        int all_Throughput = 0;

        int Throughput_avg = 0;

        int Throughput_count = 0;

        int EG_count = 0;
        
        int notEG_count = 0;

        int job_num = 0;

        int notEGtotalTasks = 0; // 总任务数（运行期根据提交的任务动态累加）

        int totaltoken = 0;

        double jobtotaltime = 0;

        double ratios = 1.0; // 超参系数

        double starvation_threshold = 0.0;

        bool enableStarvation = false;

        double totaltime = 0;

        bool istime = true;

        std::chrono::system_clock::time_point SystemstartTime;



        

        

        std::atomic<int> notEGcompletedTasks{0}; // 已完成任务数，线程安全

        struct notEGTaskStats {
    std::mutex statsMutex;
    double totalLatency[4] = {0, 0, 0, 0};
    int totalTasks[4] = {0, 0, 0, 0};
    int totalTokens[4] = {0, 0, 0, 0};

    void notEGaddResult(int TPOT, double latency, int tokens) {
        std::lock_guard<std::mutex> lock(statsMutex);
            if(TPOT == 50) {
            totalLatency[1] += latency;
                totalTasks[1]++;
                totalTokens[1] += tokens;   
            }
            else if(TPOT == 100) {
                totalLatency[2] += latency;
                totalTasks[2]++;
                totalTokens[2] += tokens;
            }
            else if(TPOT == 250) {
                totalLatency[3] += latency;
                totalTasks[3]++;
                totalTokens[3] += tokens;
            }
    }

    void notEGprintStats() {
        std::lock_guard<std::mutex> lock(statsMutex);
        std::cout << "\nadd by zp notEG================== EageServe 测试结果统计 ==================\n";
        std::cout << std::setw(15) << "任务类型" << std::setw(12) << "任务数" << std::setw(15) << "平均时间(毫秒)" << std::setw(12) <<"总时间(毫秒)" << std::setw(12) << "总Token数\n";
        const char* taskTypes[] = {"实时短任务", "实时长任务", "非实时短任务", "非实时长任务"};
        for (int i = 0; i < 4; i++) {
            double avgLatency = totalTasks[i] > 0 ? totalLatency[i] / totalTasks[i] : 0;
            double totalTime = totalLatency[i];
            std::cout << std::setw(15) << taskTypes[i] << std::setw(12) << totalTasks[i] 
                      << std::setw(15) << std::fixed << std::setprecision(2) << avgLatency
                      << std::setw(12) << totalTime << std::setw(12) << totalTokens[i] << "\n";
        }
        std::cout << "==================================================\n";
    }
   };

    notEGTaskStats notEGstats;




        private:
        
        
        bool delayScheduling = false; // 延迟调度标志
        std::condition_variable schedulerCV; // 调度触发条件变量
        int notEGtokencount = 0;
        int tokencountindex = 0;
        int tokencount = 0;




        

        
    };
}

#endif //FASTLLM_BASELLM_H