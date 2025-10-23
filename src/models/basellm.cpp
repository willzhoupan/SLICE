//
// Created by huangyuyang on 6/25/23.
//

#include "basellm.h"
#include "utils.h"
#include <sstream>
#include <cstring>
#include <fstream>
#include <chrono>
#include <algorithm>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    int ResponseContextDict::CreateHandle() {
        locker.lock();
        int newId = 0;
        while (dicts.find(newId) != dicts.end()) {
            newId++;
        }
        dicts[newId] = new ResponseContext();
        locker.unlock();
        return newId;
    }

    ResponseContext *ResponseContextDict::GetHandle(int handleId) {
        locker.lock();
        ResponseContext *ret = dicts.find(handleId) != dicts.end() ? dicts[handleId] : nullptr;
        locker.unlock();
        return ret;
    }

    void ResponseContextDict::RemoveHandle(int handleId) {
        locker.lock();
        if (dicts.find(handleId) != dicts.end()) {
            delete dicts[handleId];
            dicts.erase(handleId);
        }
        locker.unlock();
    }

    void ResponseContext::Init(int blocks, DataType dataType) {
        pastKeyValues.clear();
        for (int i = 0; i < blocks; i++) {
            pastKeyValues.push_back(std::make_pair(Data(dataType),
                                                   Data(dataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }
        intParams.clear();
        currentTokens.clear();
        while (resultTokenQueue.size() > 0){
            resultTokenQueue.pop();
        }
        isEnding = false;
        preTokens = 0;
    }

    PastKVCacheMemory::PastKVCacheMemory(const std::vector <int> &inputToken, int tokens, long long flushTime, std::vector<std::pair<Data, Data> > *kv) {
        this->inputToken = inputToken;
        this->tokens = tokens;
        this->flushTime = flushTime;
        this->recordTimes = 1;
        auto dataType = (*kv)[0].first.dataType;
        for (int i = 0; i < kv->size(); i++) {
            this->kv.push_back(std::make_pair(Data(dataType), Data(dataType)));
        }
        for (int i = 0; i < kv->size(); i++) {
            this->kv[i].first.CopyFrom((*kv)[i].first);
            this->kv[i].second.CopyFrom((*kv)[i].second);
        }
    }

    void PastKVCacheManager::SetMaxRecordNum(int maxRecordNum) {
        std::lock_guard <std::mutex> lock(this->locker);
        this->maxRecordNum = maxRecordNum;
    }

    void PastKVCacheManager::Record(const std::vector <int> &inputToken, int tokens, std::vector<std::pair<Data, Data> > *kv) {
        std::lock_guard <std::mutex> lock(this->locker);
        if (this->memorys.find(inputToken) != this->memorys.end()) {
            this->memorys[inputToken]->recordTimes++;
            this->memorys[inputToken]->flushTime = ++flushTime;
            return;
        }

        if (this->memorys.size() >= this->maxRecordNum) {
            std::vector <int> eraseToken;
            long long minFlushTime = (1LL << 60);
            for (auto &it : this->memorys) {
                if (it.second->flushTime < minFlushTime) {
                    minFlushTime = it.second->flushTime;
                    eraseToken = it.first;
                }
            }
            delete this->memorys[eraseToken];
            this->memorys.erase(this->memorys.find(eraseToken));
        }

        this->memorys[inputToken] = new PastKVCacheMemory(inputToken, tokens, ++flushTime, kv);
    }

    void PastKVCacheManager::Remove(const std::vector <int> &inputToken) {
        std::lock_guard <std::mutex> lock(this->locker);
        if (this->memorys.find(inputToken) != this->memorys.end()) {
            if ((--this->memorys[inputToken]->recordTimes) <= 0) {
                delete this->memorys[inputToken];
                this->memorys.erase(this->memorys.find(inputToken));
            }
        }
    }

    PastKVCacheMemory *PastKVCacheManager::Get(const std::vector <int> &inputToken) {
        std::lock_guard <std::mutex> lock(this->locker);
        std::vector <int> maxToken;
        for (auto &it : this->memorys) {
            const std::vector <int> &cur = it.first;
            if (cur.size() > maxToken.size() && cur.size() <= inputToken.size()) {
                bool match = true;
                for (int i = 0; i < cur.size(); i++) {
                    if (inputToken[i] != cur[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    maxToken = cur;
                }
            }
        }
        if (maxToken.size() == 0) {
            return nullptr;
        }
        this->memorys[maxToken]->flushTime = ++this->flushTime;
        return this->memorys[maxToken];
    }

    void PastKVCacheManager::Unlock() {
        locker.unlock();
    }

    basellm::~basellm() {
        dictLocker.lock();
        this->isFree = true;
        dictLocker.unlock();
        dictCV.notify_all();
        this->weight.ReleaseWeight();
    }

    std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                basellm::GetTensorMap(const std::vector <std::string> &tensorNames) {
        std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
        for (auto &name : tensorNames) {
            WeightType weightType = this->weight.GetWeightType(name);
            DataType dataType = DataType::DATA_AUTO_NONE;
            if (weightType == WeightType::LINEAR) {
                dataType = DataType::DATA_AUTO_LINEAR;
            } else if (weightType == WeightType::EMBEDDING) {
                dataType = DataType::DATA_AUTO_EMBEDDING;
            }
            ret[name].push_back(std::make_pair(name, dataType));
        }
        return ret;
    }

    std::string basellm::Response(const std::string &oriInput, RuntimeResult retCb,
                                  const fastllm::GenerationConfig &generationConfig) {
        std::string input = oriInput;
        if (this->saveHistoryChat) {
            if (lastKeyValues != nullptr) {
                if (input.size() < lastPrompt.size() || (input.substr(0, lastPrompt.size()) != lastPrompt)) {
                    lastPrompt = "";
                    lastPromptTokens = 0;
                    delete lastKeyValues;
                    lastKeyValues = nullptr;
                } else {
                    input = input.substr(lastPrompt.size());
                }
            }
        } else {
            lastPrompt = "";
            lastPromptTokens = 0;
            delete lastKeyValues;
            lastKeyValues = nullptr;
        }

        //printf("lastPrompt = %s\n", lastPrompt.c_str());
        //printf("input = %s\n", input.c_str());

#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
        std::string prompt = input;
#ifdef PY_API
        size_t pos = input.rfind("time_stamp:");
        prompt = (generationConfig.enable_hash_id && pos != -1) ? input.substr(0, pos) : input;
        size_t hash_id = std::hash<std::string>{}(input);
#endif
        Data inputIds, attentionMask, positionIds;

        Data inputTokenData = this->weight.tokenizer.Encode(prompt);
        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(1);
        for (int i = 0; i < inputTokenData.Count(0); i++) {
            inputTokens[0].push_back(((float *) inputTokenData.cpuData)[i]);
        }
        
        if (lastKeyValues == nullptr) {
            lastKeyValues = new std::vector<std::pair<Data, Data> >();
            for (int i = 0; i < block_cnt; i++) {
                lastKeyValues->push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
                lastKeyValues->back().first.SetKVCache();
                lastKeyValues->back().second.SetKVCache();
            }
        }

        std::vector<std::pair<Data, Data> > &pastKeyValues = (*lastKeyValues);
        std::string retString = "";
        std::vector<float> results;
        LastTokensManager tokens(1, generationConfig.last_n);
        int promptLen = lastPromptTokens + inputTokens[0].size(), index = 0;
        int add_special_tokens = generationConfig.add_special_tokens? 1: 0;
        FillLLMInputs(inputTokens, {{"promptLen", promptLen}, {"index", index}, {"add_special_tokens", add_special_tokens}},
                      inputIds, attentionMask, positionIds);
        ToDataType(attentionMask, this->dataType);
        while (true) {
            auto st = std::chrono::system_clock::now();
            int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, tokens);        
            tokens.units[0].Push(ret);
            if (ret == eos_token_id
                || generationConfig.stop_token_ids.find(ret) != generationConfig.stop_token_ids.end()
                || eos_token_ids.find(ret) != eos_token_ids.end()) {
                break;
            }

            results.push_back(ret);
            std::string curString = weight.tokenizer.Decode(
                    Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
            retString += curString;
            if (retCb)
#ifdef PY_API
            {
                if (generationConfig.enable_hash_id) {
                    std::stringstream ss;
                    ss << retString << "hash_id:"<<hash_id;
                    retCb(index, pybind11::bytes(ss.str()));
                } else {
                    retCb(index, pybind11::bytes(retString));
                }
            }
#else
                retCb(index, curString.c_str());
#endif
            index++;
            fflush(stdout);
            results.clear();

            inputTokens[0] = std::vector<float> {(float)ret};
            FillLLMInputs(inputTokens, {{"promptLen", promptLen}, {"index", index}, {"add_special_tokens", add_special_tokens}},
                          inputIds, attentionMask, positionIds);
            ToDataType(attentionMask, this->dataType);
            if (index == generationConfig.output_token_limit) {
                break;
            }
            // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));
        }
        if (retCb)
#ifdef PY_API
        {
            if(generationConfig.enable_hash_id){
                std::stringstream ss;
                ss << retString << "hash_id:"<<hash_id;
                retCb(-1, pybind11::bytes(ss.str()));
            }else{
                retCb(-1, pybind11::bytes(retString));
            }
        }
#else
            retCb(-1, retString.c_str());
#endif

        lastPrompt += (input + retString);
        lastPromptTokens = promptLen + index;
        return retString;
    }

    void basellm::ResponseBatch(const std::vector<std::string> &inputs, std::vector<std::string> &outputs,
                                RuntimeResultBatch retCb, const fastllm::GenerationConfig &generationConfig) {
#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
#ifdef PY_API
        std::vector<std::string> prompts;
        std::vector < size_t > hash_ids;
        for (auto _input: inputs){
            size_t hash_id = std::hash<std::string>{}(_input);
            hash_ids.push_back(hash_id);

            size_t pos = _input.rfind("time_stamp:");
            std::string prompt = (generationConfig.enable_hash_id && pos != -1) ? _input.substr(0, pos) : _input;
            prompts.push_back(prompt);
        }
#else
        std::vector<std::string> prompts = inputs;
#endif
        // 1. first
        Data inputIds, attentionMask, positionIds;

        int batch = prompts.size();
        outputs.clear();
        outputs.resize(batch, "");

        std::vector<std::vector<float> > inputTokens;
        inputTokens.resize(batch);

        for (int i = 0; i < batch; i++) {
            Data now = this->weight.tokenizer.Encode(prompts[i]);
            for (int j = 0; j < now.Count(0); j++) {
                inputTokens[i].push_back(((float *) now.cpuData)[j]);
            }
        }

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(dataType),
                                                   Data(dataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }

        std::vector <std::map <std::string, int> > params;
        params.resize(batch);
        for (int i = 0; i < batch; i++) {
            params[i]["promptLen"] = (int)inputTokens[i].size();
        }
        params[0]["index"] = 0;
        int index = 0;
        params[0]["add_special_tokens"] = generationConfig.add_special_tokens? 1: 0;

        LastTokensManager tokensManager (batch, generationConfig.last_n);
        std::vector <bool> isEnding = std::vector <bool> (batch, false);
        FillLLMInputsBatch(inputTokens, params, inputIds, attentionMask, positionIds);
        ToDataType(attentionMask, this->dataType);
        while (true) {
            auto st = std::chrono::system_clock::now();
// ClearProfiler();
            std::vector <int> ret = ForwardBatch(batch, inputIds, attentionMask, positionIds, pastKeyValues,
                                                 generationConfig, tokensManager);
            
// PrintProfiler();
            for (int i = 0; i < batch; i++) {
                tokensManager.units[i].Push(ret[i]);
            }
            std::vector <float> fret;
            std::vector <float> results;
            int endingCount = 0;
            std::vector <std::string> curStrings;
            for (int i = 0; i < batch; i++) {
                fret.push_back(ret[i]);
                inputTokens[i] = std::vector <float> {(float)ret[i]};
                if (ret[i] == eos_token_id || eos_token_ids.find(ret[i]) != eos_token_ids.end()) {
                    isEnding[i] = true;
                } else {
                    auto itStopTk = generationConfig.stop_token_ids.find(ret[i]);
                    if (itStopTk != generationConfig.stop_token_ids.end()) {
                        isEnding[i] = true;
                    }
                }
                if (isEnding[i]) {
                    curStrings.push_back("");
                    endingCount++;
                    continue;
                }
                results.push_back(ret[i]);
                std::string curString = weight.tokenizer.Decode(
                        Data(DataType::FLOAT32, {(int) results.size()}, results)).c_str();
                outputs[i] += curString;
                curStrings.push_back(curString);
                results.clear();
            }

            if (endingCount == batch) {
                break;
            }
            if (retCb)
#ifdef PY_API
            {
                if (generationConfig.enable_hash_id) {
                    std::vector<pybind11::bytes> rtnStrings;
                    for (size_t i=0; i<batch; i++){
                        std::stringstream ss;
                        ss << curStrings[i] << "hash_id:" << hash_ids[i];
                        rtnStrings.push_back(pybind11::bytes(ss.str()));
                    }
                    retCb(index, rtnStrings);
                } else {
                    std::vector<pybind11::bytes> rtnStrings;
                    for (size_t i=0; i<batch; i++){
                        std::stringstream ss;
                        ss << curStrings[i];
                        rtnStrings.push_back(pybind11::bytes(ss.str()));
                    }
                    retCb(index, rtnStrings);
                }
            }
#else
                retCb(index, curStrings);
#endif
            index++;
            params[0]["index"] = index;
            FillLLMInputsBatch(inputTokens, params, inputIds, attentionMask, positionIds);
            ToDataType(attentionMask, this->dataType);
            // printf("len = %d, spend %f s.\n", len, GetSpan(st, std::chrono::system_clock::now()));

            if (index == generationConfig.output_token_limit) {
                break;
            }
        }
        if (retCb)
#ifdef PY_API
        {
            if (generationConfig.enable_hash_id) {
                std::vector<pybind11::bytes> rtnStrings;
                for (size_t i=0; i<batch; i++){
                    std::stringstream ss;
                    ss << outputs[i] << "hash_id:" << hash_ids[i];
                    rtnStrings.push_back(pybind11::bytes(ss.str()));
                }
                retCb(-1, rtnStrings);
            } else {
                std::vector<pybind11::bytes> rtnStrings;
                for (size_t i=0; i<batch; i++){
                    std::stringstream ss;
                    ss << outputs[i];
                    rtnStrings.push_back(pybind11::bytes(ss.str()));
                }
                retCb(-1, rtnStrings);
            }
        }
#else
            retCb(-1, outputs);
#endif
    }

    std::vector<int> basellm::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                           const fastllm::Data &positionIds,
                                           std::vector<std::pair<Data, Data>> &pastKeyValues,
                                           const fastllm::GenerationConfig &generationConfig,
                                           const fastllm::LastTokensManager &lastTokens,
                                           std::vector <std::vector <float>*> *retLogits) {
        printf("Unsupport forward batch.\n");
        exit(0);
    }

    std::vector<int> basellm::ForwardBatch(int batch, const fastllm::Data &inputIds,
                                           const std::vector<Data *> &attentionMask,
                          const std::vector<Data *> &positionIds, const std::vector<int> &seqLens,
                          std::vector<std::pair<Data *, Data *>> &pastKeyValues,
                          const std::vector<GenerationConfig> &generationConfigs,
                          const fastllm::LastTokensManager &lastTokens,
                          std::vector <std::vector <float>*> *logits,
                          bool enableSLICE,
                          int tokencount) {
        std::vector <int> ret;
        int cur = 0;
        for (int i = 0; i < batch; i++) {
            std::vector<std::pair<Data, Data> > curKV;
            curKV.resize(this->block_cnt);
            for (int j = 0; j < this->block_cnt; j++) {
                Mul(*pastKeyValues[i * this->block_cnt + j].first, 1.0, curKV[j].first);
                Mul(*pastKeyValues[i * this->block_cnt + j].second, 1.0, curKV[j].second);
            }
            Data curInput;
            Split(inputIds, 1, cur, cur + seqLens[i], curInput);
            LastTokensManager curTokens;
            curTokens.units.push_back(lastTokens.units[i]);
            ret.push_back(this->Forward(curInput, *attentionMask[i], *positionIds[i], curKV, generationConfigs[i], curTokens));
            for (int j = 0; j < this->block_cnt; j++) {
                Mul(curKV[j].first, 1.0, *pastKeyValues[i * this->block_cnt + j].first);
                Mul(curKV[j].second, 1.0, *pastKeyValues[i * this->block_cnt + j].second);
            }
        }
        return ret;
    }

    int basellm::LaunchResponseTokens(const std::vector<int> &inputTokens,
                                      const fastllm::GenerationConfig &generationConfig,
                                      double ratios,
                                      int batch) {
        this->maxBatch = batch;
        this->ratios = ratios;
        mainLoopLocker.lock();
        if (mainLoop == nullptr) {
            if (mainLoop == nullptr) {
                mainLoop = new std::thread([this](basellm *model) {
                    long long kvCacheLimit = 16LL << 30;
#ifdef USE_CUDA
                    auto freeSizes = FastllmCudaGetFreeSizes();
                    auto dmap = GetDeviceMap();
                    std::set <int> deviceIds;
                    for (auto &it : dmap) {
                        if (StartWith(it.first, "cuda")) {
                            for (int id : ParseDeviceIds(it.first, "cuda")) {
                                deviceIds.insert(id);
                            }
                        }
                    }
                    if (deviceIds.size() == 0) {
                        deviceIds.insert(0);
                    }
                    kvCacheLimit = 0;
                    for (int id : deviceIds) {
                        kvCacheLimit += std::max(0LL, freeSizes[id] - (2LL << 30));
                    }
#endif
                    if (model->kvCacheLimit > 0) {
                        kvCacheLimit = model->kvCacheLimit;
                    }

                    int unitSize = (model->dataType == DataType::FLOAT32 ? 4 : 2);
                    int maxTotalLens = kvCacheLimit / (model->elementsInKVCachePerToken * unitSize);
                    if (model->elementsInKVCachePerToken <= 0) {
                        maxTotalLens = kvCacheLimit / 1024 / 1024;
                    }
                    if (model->tokensLimit > 0) {
                        maxTotalLens = model->tokensLimit;
                    }

                    int maxBatch = std::max(1, std::min(512, maxTotalLens / 128));
                    if (model->maxBatch > 0) {
                        maxBatch = model->maxBatch;
                    }
                    
                    model->tokensLimit = maxTotalLens;
                    printf("maxTotalLens:%d\n",maxTotalLens);
                    int limit = maxTotalLens;
                    model->promptLimit = limit * 2 / 3;

                    if (model->verbose) {
                        printf("Fastllm KV Cache Limit: %f MB.\n", (double)kvCacheLimit / 1024 / 1024);
                        printf("Fastllm KV Cache Token limit: %d tokens.\n", maxTotalLens);
                        printf("Fastllm Prompt Token limit: %d tokens.\n", std::min(model->max_positions, model->promptLimit));
                        printf("Fastllm Batch limit: %d.\n", maxBatch);
                        if (model->enableSLICE) {
                            printf("SLICE Scheduler enabled.\n");
                        }
                    }

                    auto lastRecordTime = std::chrono::system_clock::now();
                    long long genTokens = 0;

                    while (true) {
                        if (model->isFree) {
                            break;
                        }

                        std::vector <Data*> attentionMasks;
                        std::vector <Data*> positionIds;
                        std::vector <std::pair <Data*, Data*> > pastKeyValues;
                        std::vector <float> ids;
                        std::vector <int> seqLens;
                        std::vector <int> handles;
                        std::vector <GenerationConfig> generationConfigs;
                        LastTokensManager tokensManager;
                        std::vector <std::vector <float>* > logits;                     
                        std::unique_lock<std::mutex> dictLocker(model->dictLocker); 
                        // 如果处于延迟调度模式，则不唤醒后台线程
                        // model->dictCV.wait(dictLocker, [model]() {
                        // return !model->delayScheduling || 
                        // !model->responseContextDict.dicts.empty() || 
                        // model->isFree;
                        // });

                        // 首先把已经abort的请求删除掉
                        std::set <int> abortHandles;
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->isAbort) {
                                abortHandles.insert(it.first);
                            }
                        }
                        for (auto &it : abortHandles) {
                            model->responseContextDict.RemoveHandle(it);
                        }

                        int limit = maxTotalLens;
                        int promptLimit = model->promptLimit;

                        int lenSum = 0;
                        for (auto &it: model->responseContextDict.dicts) {
                            if (it.second->pastKeyValues[0].first.expansionDims.size() > 0) {
                                lenSum += it.second->pastKeyValues[0].first.expansionDims[1];
                            }
                        }

                        // 使用SLICE调度器或传统方式获取下一批任务
                        if (model->enableSLICE) {
                            try {
                                // 获取下一批任务
                                std::set<int> nextBatch;
                                
                                try {
                                    int totalTPOT = 0;
                                    totalTPOT = model->scheduler.GetNextBatch(tokencountindex, model->ratios);
                                    nextBatch = model->scheduler.currentBatch;
                                } catch (const std::exception& e) {
                                    printf("Exception in GetNextBatch: %s\n", e.what());
                                    nextBatch.clear();
                                } catch (...) {
                                    printf("Unknown exception in GetNextBatch\n");
                                    nextBatch.clear();
                                }

                                auto startfor = std::chrono::system_clock::now();
                                
                                for (int handleId : nextBatch) {
                                    try {
                                        
                                        if (model->responseContextDict.dicts.find(handleId) == model->responseContextDict.dicts.end()) {
                                            printf("Warning: HandleId %d not found in responseContextDict\n", handleId);
                                            continue;
                                        }
                                        
                                        ResponseContext* context = model->responseContextDict.GetHandle(handleId);
                                        if (!context || context->isEnding) {
                                            continue;
                                        }
                                        
                                        generationConfigs.push_back(context->generationConfig);
                                        if (context->generationConfig.output_logits) {
                                            context->resultLogits.push(new std::vector<float>());
                                            logits.push_back(context->resultLogits.back());
                                        } else {
                                            logits.push_back(nullptr);
                                        }

                                        tokensManager.units.push_back(context->tokens);
                                        handles.push_back(handleId);

                                        if (context->preTokens == 0) {
                                            context->intParams["add_special_tokens"] = context->cacheLen > 0 ? false : context->generationConfig.add_special_tokens;
                                            context->intParams["promptLen"] = context->cacheLen + context->currentTokens.size();
                                            context->intParams["index"] = 0;
                                        } else {
                                            context->intParams["index"]++;
                                        }
                                        Data inputIds, attentionMask, curPositionIds;
                                        std::vector<std::vector<float> > tokens;
                                        tokens.resize(1);
                                        for (int i: context->currentTokens) {
                                            tokens[0].push_back(i);
                                        }
                                        model->FillLLMInputs(tokens, context->intParams, inputIds, attentionMask, curPositionIds);
                                        ToDataType(attentionMask, model->dataType);

                                        seqLens.push_back(inputIds.Count(0));
                                        for (int i = 0; i < inputIds.Count(0); i++) {
                                            ids.push_back(((float *) inputIds.cpuData)[i]);
                                        }
                                        if (attentionMask.dims.size() == 0) {
                                            attentionMasks.push_back(nullptr);
                                        } else {
                                            attentionMasks.push_back(new Data());
                                            attentionMask.ToDevice(DataDevice::CPU);
                                            attentionMasks.back()->CopyFrom(attentionMask);
                                        }
                                        if (curPositionIds.dims.size() == 0) {
                                            positionIds.push_back(nullptr);
                                        } else {
                                            positionIds.push_back(new Data());
                                            positionIds.back()->CopyFrom(curPositionIds);
                                        }
                                        context->preTokens += seqLens.back();
                                        for (int i = 0; i < model->block_cnt; i++) {
                                            pastKeyValues.push_back(std::make_pair(&context->pastKeyValues[i].first,
                                                                                   &context->pastKeyValues[i].second));
                                        }
                                    } catch (const std::exception& e) {
                                        printf("Exception processing task %d: %s\n", handleId, e.what());
                                    } catch (...) {
                                        printf("Unknown exception processing task %d\n", handleId);
                                    }
                                }
                            } catch (const std::exception& e) {
                                printf("Exception in SLICE main loop: %s\n", e.what());
                                model->enableSLICE = false;  // 暂时禁用SLICE
                            } catch (...) {
                                printf("Unknown exception in SLICE main loop\n");
                                model->enableSLICE = false;  // 暂时禁用SLICE
                            }                      
                        } else {
                            for (int isPrompt = 1; isPrompt >= 0; isPrompt--) {
                                int cnt = 0;
                                if (isPrompt == 0 && seqLens.size() > 0) {
                                    continue;
                                }
                                if (lenSum >= promptLimit && isPrompt) {
                                    continue;
                                }

                                for (auto &it: model->responseContextDict.dicts) {
                                    if (it.second->isEnding) {
                                        continue;
                                    }
                                    if (isPrompt && it.second->preTokens != 0) {
                                        continue;
                                    }
                                    if (!isPrompt && it.second->preTokens == 0) {
                                        continue;
                                    }
                                    if (it.second->currentTokens.size() > promptLimit) {
                                        it.second->isEnding = true;
                                        it.second->error = ResponseContextErrorPromptTooLong;
                                        continue;
                                    }

                                    int outputLimit = it.second->generationConfig.output_token_limit;
                                    outputLimit = (outputLimit < 0 ? 128 : outputLimit);
                                    if (isPrompt && lenSum + it.second->currentTokens.size() > promptLimit) {
                                        continue;
                                    }

                                    if (!isPrompt) {
                                        if (it.second->pastKeyValues[0].first.expansionDims[1] == it.second->pastKeyValues[0].first.dims[1]) {
                                            int sur = it.second->generationConfig.output_token_limit - it.second->curTokens;                                        
                                            int predictLen = 256;
                                            if (sur > 0) {
                                                predictLen = std::min(predictLen, ((sur - 1) / 128 + 1) * 128);
                                            }
                                            if (lenSum + predictLen > limit) {
                                                continue;
                                            }
                                            lenSum += predictLen;
                                        }
                                    }

                                    generationConfigs.push_back(it.second->generationConfig);
                                    if (it.second->generationConfig.output_logits) {
                                        it.second->resultLogits.push(new std::vector<float>());
                                        logits.push_back(it.second->resultLogits.back());
                                    } else {
                                        logits.push_back(nullptr);
                                    }

                                    tokensManager.units.push_back(it.second->tokens);
                                    handles.push_back(it.first);

                                    if (it.second->preTokens == 0) {
                                        it.second->intParams["add_special_tokens"] = it.second->cacheLen > 0 ? false : it.second->generationConfig.add_special_tokens;
                                        it.second->intParams["promptLen"] = it.second->cacheLen + it.second->currentTokens.size();
                                        it.second->intParams["index"] = 0;
                                    } else {
                                        it.second->intParams["index"]++;
                                    }
                                    Data inputIds, attentionMask, curPositionIds;
                                    std::vector<std::vector<float> > tokens;
                                    tokens.resize(1);
                                    for (int i: it.second->currentTokens) {
                                        tokens[0].push_back(i);
                                    }
                                    model->FillLLMInputs(tokens, it.second->intParams, inputIds, attentionMask, curPositionIds);
                                    ToDataType(attentionMask, model->dataType);

                                    seqLens.push_back(inputIds.Count(0));
                                    for (int i = 0; i < inputIds.Count(0); i++) {
                                        ids.push_back(((float *) inputIds.cpuData)[i]);
                                    }
                                    if (attentionMask.dims.size() == 0) {
                                        attentionMasks.push_back(nullptr);
                                    } else {
                                        attentionMasks.push_back(new Data());
                                        attentionMask.ToDevice(DataDevice::CPU);
                                        attentionMasks.back()->CopyFrom(attentionMask);
                                    }
                                    if (curPositionIds.dims.size() == 0) {
                                        positionIds.push_back(nullptr);
                                    } else {
                                        positionIds.push_back(new Data());
                                        positionIds.back()->CopyFrom(curPositionIds);
                                    }
                                    it.second->preTokens += seqLens.back();
                                    for (int i = 0; i < model->block_cnt; i++) {
                                        pastKeyValues.push_back(std::make_pair(&it.second->pastKeyValues[i].first,
                                                                               &it.second->pastKeyValues[i].second));
                                    }
                                    if (isPrompt) {                                    
                                        cnt += it.second->currentTokens.size();

                                        if (cnt > 1024) {
                                            break;
                                        }
                                        // break;
                                    }

                                    if (seqLens.size() >= maxBatch || lenSum + seqLens.size() * 128 > limit) {
                                        break;
                                    }
                                }
                            }
                        }
                        

                        
                        if (seqLens.size() > 0) {

                            
                            std::vector <std::pair <Data, Data> > *pastKeyValue1;
                            if (seqLens.size() == 1) {
                                pastKeyValue1 = &model->responseContextDict.dicts[handles[0]]->pastKeyValues;
                            }
                            dictLocker.unlock();
#ifdef USE_CUDA
                            FastllmCudaClearBigBuffer();
#endif
                            Data inputIds = Data(DataType::FLOAT32, {1, (int) ids.size()}, ids);
                            std::vector<int> ret;

                            auto st = std::chrono::system_clock::now();
//ClearProfiler();
                            if (seqLens.size() > 1) {
                                
                             
                                if (model->enableSLICE) {
                                ret = model->ForwardBatch(seqLens.size(), inputIds, attentionMasks,
                                                          positionIds, seqLens, pastKeyValues, generationConfigs,
                                                          tokensManager, &logits,model->enableSLICE,model->tokencount);
                                }
                                else{
                                ret = model->ForwardBatch(seqLens.size(), inputIds, attentionMasks,
                                                          positionIds, seqLens, pastKeyValues, generationConfigs,
                                                          tokensManager, &logits,model->enableSLICE,model->notEGtokencount);
                                }
                            }
                            else {
                                if (seqLens[0] > 8192) {
                                    int len = seqLens[0];
                                    int first = 8192, part = 2048;
                                    for (int st = 0; st < len; ) {
                                        int curLen = std::min(st == 0 ? first : part, len - st);
                                        Data curInput, curPositionIds;
                                        Split(inputIds, 1, st, st + curLen, curInput);
                                        Split(*positionIds[0], 1, st, st + curLen, curPositionIds);

                                        ret = std::vector <int> {model->Forward(curInput, Data(), curPositionIds,
                                            *pastKeyValue1, generationConfigs[0], tokensManager, logits[0])};
                                        st += curLen;
                                    }
                                } else {
                                    int handleId = handles[0];
                                    auto *context = model->responseContextDict.GetHandle(handleId);
                                    if (context && context->preTokens == seqLens[0]) { // 或 context->preTokens == 0
                                        int promptTokenCount = context->cacheLen + context->currentTokens.size();
                                        auto prefillStart = std::chrono::system_clock::now();
                                        ret = std::vector <int> {model->Forward(
                                        inputIds,
                                        attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                                        *positionIds[0],
                                        *pastKeyValue1, generationConfigs[0], tokensManager, logits[0])};
                                    if (model->verbose) {
                                    auto prefillEnd = std::chrono::system_clock::now();
                                    float prefillTime = fastllm::GetSpan(prefillStart, prefillEnd);
                                    context->prefillTime = prefillTime;
                                    {
                                        std::ofstream fout("prefill_time_log.txt", std::ios::app);
                                        if (fout.is_open()) {
                                            fout << "handleId: " << handleId
                                            << ", promptTokenCount: " << promptTokenCount
                                            << ", prefillTime: " << prefillTime << " s\n";
                                            fout.close();
                                        }
                                    }
                                    }    
                                }
                                    else{
                                    
                                        ret = std::vector <int> {model->Forward(inputIds,
                                            attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                                            *positionIds[0],
                                            *pastKeyValue1, generationConfigs[0], tokensManager, logits[0])};
                                    
                                    }

                                }
                            }

                            auto endTime = std::chrono::system_clock::now();
                            float batchTime = fastllm::GetSpan(st, endTime);
                            int totalTokens = seqLens.size(); // 这一批生成的token数量

                            if (batchTime > 0) {
                                float tokensPerSecond = totalTokens / batchTime;

                                if (model->verbose) {
                                // 控制台输出
                                printf("[BATCH DEBUG] 批次大小: %d, 耗时: %.4f s, 速度: %.2f tokens/s\n", 
                                    (int)seqLens.size(), batchTime, tokensPerSecond);
                                
                                // 文件输出
                                std::ofstream batchFile("batch_token_speed.txt", std::ios::app);
                                if (batchFile.is_open()) {
                                    batchFile << "===============系统吞吐量大小统计==================\n";
                                    batchFile << "时间戳: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                                        endTime.time_since_epoch()).count() << " ms, "
                                            << "批次大小: " << seqLens.size() << ", "
                                            << "耗时: " << batchTime << " s, "
                                            << "速度: " << tokensPerSecond << " tokens/s\n";
                                    batchFile.close();
                                }
                              }
                            }
                            

// PrintProfiler();
// int total = 0;
// for (int i : seqLens) total += i;
// float spend = GetSpan(st, std::chrono::system_clock::now());
// printf("len = %d, spend = %f s. tokens / s = %f\n", (int)total, spend, (float)total / spend);

                            if (model->verbose) {
                                int total = 0;
                                for (int i : seqLens) total += i;
                                float spend = GetSpan(st, std::chrono::system_clock::now());
                                printf("len = %d, spend = %f s. tokens / s = %f\n", (int)total, spend, (float)total / spend);
                            }
                            
                            dictLocker.lock();
                            
                            for (int i = 0; i < handles.size(); i++) {
                                auto it = model->responseContextDict.dicts.find(handles[i]);
                                if (it == model->responseContextDict.dicts.end()) {
                                    printf("Warning: Handle %d not found in responseContextDict\n", handles[i]);
                                    continue;
                                }
                                
                                int curRet = ret[i];

                                // 更新已生成的token数量
                                it->second->tokenGenerated++;
                                model->notEGtokencount++;

                                if (it->second->curTokens == 2 && it->second->isrecord == true)
                                {
                                    it->second->slicetime = std::chrono::system_clock::now();
                                    it->second->isrecord = false;
                                } 
                                
                                
                                if (curRet == model->eos_token_id || model->eos_token_ids.find(curRet) != model->eos_token_ids.end()) {
                                    it->second->isEnding = true;
                                    if (model->verbose) {
                                    auto endTime = std::chrono::system_clock::now();
                                    auto startTime = it->second->notEGstartTime;
                                    float duration = fastllm::GetSpan(startTime, endTime);
                                    float sliceduration = fastllm::GetSpan(it->second->slicetime, endTime);
                                    int TPOT = it->second->TPOT; 
                                    int priority = it->second->priority;
                                    int tokens = it->second->curTokens;
                                    model->totaltoken += tokens;
                                    this->notEGstats.notEGaddResult(TPOT, duration, tokens);
                                    model->totaltime += duration;
                                    model->jobtotaltime += sliceduration;

                                    // 计算平均单token延迟(ms)，并判断是否满足SLO(=TPOT, ms)
                                    double avgTokenMs = 0.0;
                                    bool satisfySLO = false;
                                    if (tokens > 0) {
                                        avgTokenMs = (sliceduration * 1000.0) / tokens; // ms / token
                                        satisfySLO = (avgTokenMs <= static_cast<double>(it->second->TPOT));
                                    }
                                    it->second->satisfySLO = satisfySLO;

                                    std::cout << "add by zp  任务HandleId #" << model->notEGcompletedTasks << " (" 
                                    << (it->second->isRealtime ? "实时" : "非实时") << (it->second->isShortTask ? "短" : "长") 
                                    << "任务) 完成，延迟: " << duration << "  s, 生成 " 
                                    << it->second->curTokens << " 个tokens, 平均单token延迟: " << avgTokenMs << " ms, TPOT: "
                                    << it->second->TPOT << " ms, 满足SLO: " << (it->second->satisfySLO ? "是" : "否") << "\n";

                                    std::ofstream fout("JobThroughPutlog.txt", std::ios::app);
                                    if (fout.is_open()) {
                                                fout << "=========================================\n";      
                                                fout << "handleId: " << handles[i]
                                                     << ", Latency: " << sliceduration << " s"
                                                     << ", Generated Token: " << it->second->curTokens << " tokens"
                                                     << ", ThroughPut: " << it->second->curTokens / sliceduration << " tokens/s"
                                                     << ", TPOT: " << it->second->TPOT << " ms"
                                                     << ", AvgTokenLatency: " << avgTokenMs << " ms"
                                                     << ", satisfySLO: " << (it->second->satisfySLO ? "true" : "false") << "\n";
                                                     
                                                fout.close();
                                    }
                                    }
                                    model->notEGcompletedTasks++;

                                    try {
                                        if (model->enableSLICE) {
                                            model->scheduler.TaskCompleted(handles[i]);
                                        }
                                    } catch (...) {
                                        printf("Exception in TaskCompleted for handle %d\n", handles[i]);
                                    }
                                } else {
                                    auto itStopTk = it->second->generationConfig.stop_token_ids.find(curRet);
                                    if (itStopTk != it->second->generationConfig.stop_token_ids.end()) {
                                            it->second->isEnding = true;
                                            try {
                                                if (model->enableSLICE) {
                                                    model->scheduler.TaskCompleted(handles[i]);
                                                }
                                            } catch (...) {
                                                printf("Exception in TaskCompleted for handle %d\n", handles[i]);
                                            }
                                    }
                                }

                                if (it->second->isEnding == false) {
                                    it->second->currentTokens = std::vector<int>{curRet};
                                    it->second->resultTokenQueue.push(curRet);

                                    it->second->tokens.Push(curRet);
                                    it->second->curTokens++;
                                    if (model->verbose) {
                                    // 计算TTFT
                                    if (!it->second->ttf_recorded && it->second->curTokens == 2) {
                                        auto ttf_end = std::chrono::system_clock::now();
                                        it->second->ttf_time = fastllm::GetSpan(it->second->ttf_start_time, ttf_end);
                                        it->second->ttf_recorded = true;
                                        {
                                            std::ofstream fout("ttft_log.txt", std::ios::app);
                                            if (fout.is_open()) {
                                                fout << "handleId: " << handles[i]
                                                    << ", request TPOT: " << it->second->TPOT << " ms"
                                                     << ", TTFT: " << it->second->ttf_time << " s\n";
                                                fout.close();
                                            }
                                        }
                                      }
                                    }
                                    


                                    if (it->second->curTokens == it->second->generationConfig.output_token_limit) {
                                        it->second->isEnding = true;
                                        try {
                                            if (model->enableSLICE) {
                                                model->scheduler.TaskCompleted(handles[i]);
                                            }
                                        } catch (...) {
                                            printf("Exception in TaskCompleted for handle %d\n", handles[i]);
                                        }
                                    }
                                }
                            }
                        } 




                        for (int i = 0; i < attentionMasks.size(); i++) {
                            delete attentionMasks[i];
                        }
                        for (int i = 0; i < positionIds.size(); i++) {
                            delete positionIds[i];
                        }

                        if (seqLens.size() == 0) {
                            // 当没有任务时，等待新任务到达
                            if (model->enableSLICE) {
                                // 检查是否还有任务在队列中
                                bool hasTasks = false;
                                for (int priority = 0; priority < 4; priority++) {
                                    if (!model->scheduler.priorityQueues[priority].empty()) {
                                        hasTasks = true;
                                        break;
                                    }
                                }
                                if (!hasTasks && model->responseContextDict.dicts.empty()) {
                                    // 如果队列为空且没有正在处理的任务，等待新任务
                                    model->dictCV.wait(dictLocker);
                                }
                            } else {
                                model->dictCV.wait(dictLocker);
                            }
                        }
                    }
                }, this);
            }
        }
        mainLoopLocker.unlock();


        
        dictLocker.lock();
        int handleId = -1;
        static int job_num = 0;
        try {
            handleId = responseContextDict.CreateHandle();
            ResponseContext *context = responseContextDict.GetHandle(handleId);
            if (!context) {
                printf("Critical error: failed to create context for handle %d\n", handleId);
                dictLocker.unlock();
                return -1;
            }
            
            try {
                context->Init(this->block_cnt, this->dataType);
            } catch (const std::exception& e) {
                printf("Exception in context initialization: %s\n", e.what());
                responseContextDict.RemoveHandle(handleId);
                dictLocker.unlock();
                return -1;
            } catch (...) {
                printf("Unknown exception in context initialization\n");
                responseContextDict.RemoveHandle(handleId);
                dictLocker.unlock();
                return -1;
            }

            if (this->istime) {
                this->SystemstartTime = std::chrono::system_clock::now();
                this->istime = false;
            }
            

            context->Init(this->block_cnt, this->dataType);
            context->currentTokens = inputTokens;
            context->generationConfig = generationConfig;
            context->tokens = LastTokensUnit(generationConfig.last_n);
            context->isRealtime = generationConfig.isRealtime;
            context->isShortTask = generationConfig.isShortTask;
            context->priority = generationConfig.priority;
            context->tokenGenerated = 0;
            context->notEGstartTime = std::chrono::system_clock::now();
            context->ttf_start_time = std::chrono::system_clock::now();
            context->ttf_time = 0.0f;

            // 任务成功提交，累计总任务数
            this->notEGtotalTasks++;
            context->ttf_recorded = false;
            context->reward = generationConfig.reward;
            context->TPOT = generationConfig.TPOT;
            


            try {
                auto cache = pastKVCacheManager.Get(inputTokens);
                if (cache != nullptr) {
                    for (int i = 0; i < this->block_cnt; i++) {
                        context->pastKeyValues[i].first.CopyFrom(cache->kv[i].first);
                        context->pastKeyValues[i].second.CopyFrom(cache->kv[i].second);
                    }
                    context->currentTokens.erase(context->currentTokens.begin(), context->currentTokens.begin() + cache->inputToken.size());
                    context->cacheLen = cache->inputToken.size();
                }
            } catch (const std::exception& e) {
                printf("Exception in cache handling: %s\n", e.what());
                // 不严重，可以继续执行
            } catch (...) {
                printf("Unknown exception in cache handling\n");
                // 不严重，可以继续执行
            }

            // 如果启用了SLICE，将任务添加到调度器
            bool SLICESuccess = false;
            try {
                if (enableSLICE) {

                    // 添加到调度器
                    try {
                        int u = 0;
                        if (context->TPOT == 50) {
                            u = 100;
                        }
                        else  {
                            u = 1;
                        }
                        scheduler.AddTask(handleId, context, context->isRealtime, context->isShortTask,context->priority,context->reward,context->TPOT , u);
                        printf("handleId: %d, isRealtime: %d, isShortTask: %d, priority: %d, reward: %d, TPOT: %d\n", handleId, context->isRealtime, context->isShortTask, context->priority, context->reward, context->TPOT);
                        SLICESuccess = true;                                    
                    } catch (const std::exception& e) {
                        printf("SLICE AddTask exception: %s\n", e.what());
                        // 出现异常时暂时禁用SLICE
                        enableSLICE = false;
                    } catch (...) {
                        printf("SLICE AddTask unknown exception\n");
                        // 出现异常时暂时禁用SLICE
                        enableSLICE = false;
                    }
                }
            } catch (...) {
                printf("SLICE related exception in LaunchResponseTokens\n");
                // 出现异常时暂时禁用SLICE
                enableSLICE = false;
            }

            // 确保我们能告知调用者任务是否成功添加到SLICE
            if (verbose && enableSLICE) {
                if (SLICESuccess) {
                    printf("Task %d successfully added to SLICE scheduler\n", handleId);
                    job_num++;
                    
                    
                } else {
                    printf("Task %d failed to add to SLICE scheduler, falling back to traditional scheduling\n", handleId);
                }
            }



            dictLocker.unlock();

            if(!delayScheduling) {
                dictCV.notify_one();
            }
     
            return handleId;

        } catch (const std::exception& e) {
            printf("Exception in LaunchResponseTokens: %s\n", e.what());
            if (handleId >= 0) {
                responseContextDict.RemoveHandle(handleId);
            }
            dictLocker.unlock();
            return -1;
        } catch (...) {
            printf("Unknown exception in LaunchResponseTokens\n");
            if (handleId >= 0) {
                responseContextDict.RemoveHandle(handleId);
            }
            dictLocker.unlock();
            return -1;
        }
    }



    void basellm::StartScheduler() {
        {
       std::lock_guard<std::mutex> lock(dictLocker);
       delayScheduling = false;
        }
       dictCV.notify_all(); // 唤醒所有等待的线程
       }


    bool basellm::CanFetchResponse(int handleId) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            return true;
        } else {
            return (context->resultTokenQueue.size() > 0 || context->isEnding);
        }
    }

    void basellm::AbortResponse(int handleId) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        
        if (context == nullptr) {
            return;
        } else {
            context->isAbort = true;
        }
    }
    
    int basellm::FetchResponseTokens(int handleId) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            return -1;
        } else {
            static auto lastTokenTime = std::chrono::system_clock::now();
            static int debugtokenCount = 0;

            while (true) {
                if (context->resultTokenQueue.size() > 0) {
                    int ret = context->resultTokenQueue.front();
                    context->resultTokenQueue.pop();
                    
                    return ret;
                } else {
                    if (context->isEnding) {
                        if (enableSLICE) {
                            printf("任务完成 输出token为：%d\n",context->curTokens);
                        }
                        responseContextDict.RemoveHandle(handleId);
                        dictLocker.unlock();
                        dictCV.notify_one();
                        if (context->error == ResponseContextErrorNone) {
                            return -1;
                        } else if (context->error == ResponseContextErrorPromptTooLong) {
                            return -2;
                        } else {
                            return -1;
                        }
                    }
                }
                dictLocker.unlock();
                MySleep(0);
                dictLocker.lock();
            }
        }
    }



    int basellm::FetchResponseLogits(int handleId, std::vector<float> &logits) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        ResponseContext *context = responseContextDict.GetHandle(handleId);
        if (context == nullptr) {
            return -1;
        } else {
            while (true) {
                if (context->resultTokenQueue.size() > 0) {
                    int ret = context->resultTokenQueue.front();
                    context->resultTokenQueue.pop();
                    if (!context->resultLogits.empty()) {
                        logits = *context->resultLogits.front();
                        delete context->resultLogits.front();
                        context->resultLogits.pop();
                    }
                    return ret;
                } else {
                    if (context->isEnding) {
                        responseContextDict.RemoveHandle(handleId);
                        dictLocker.unlock();
                        dictCV.notify_one();
                        return -1;
                    }
                }
                dictLocker.unlock();
                MySleep(0);
                dictLocker.lock();
            }
        }
    }

    void basellm::AddPromptCache(const std::vector <int> &inputTokens) {
        std::unique_lock<std::mutex> dictLocker(this->dictLocker);
        auto cache = pastKVCacheManager.Get(inputTokens);
        if (cache != nullptr && cache->inputToken.size() == inputTokens.size()) {
            return;
        }
        Data inputIds, attentionMask, positionIds;
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }

        int promptLen = inputTokens.size(), index = 0;
        int add_special_tokens = false;
        std::vector <std::vector <float> > fInputTokens;
        fInputTokens.resize(1);
        for (int i = 0; i < inputTokens.size(); i++) {
            fInputTokens[0].push_back(inputTokens[i]);
        }
        FillLLMInputs(fInputTokens, {{"promptLen", promptLen}, {"index", index}, {"add_special_tokens", add_special_tokens}},
                      inputIds, attentionMask, positionIds);
        ToDataType(attentionMask, this->dataType);
        int ret = Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        pastKVCacheManager.Record(inputTokens, inputTokens.size(), &pastKeyValues);
    }

    bool basellm::NeedAttentionMask(int qlen, int klen) {
        return true;
    }

    // 根据输入的tokens生成LLM推理的输入
    void basellm::FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                               const std::map <std::string, int> &params,
                               Data &inputIds, Data &attentionMask, Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int index = params.find("index")->second;
        int promptLen = params.find("promptLen")->second;

        if (inputTokens[0].size() > 1) {
            int seqLen = inputTokens[0].size();
            std::vector <float> vpids = std::vector <float> (seqLen, 0);
            for (int i = 0; i < seqLen; i++) {
                vpids[i] = promptLen - seqLen + i;
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vpids));
            
            if (NeedAttentionMask(seqLen, promptLen)) {
                std::vector <float> vmask = std::vector <float> (seqLen * promptLen, 0);
                for (int i = 0; i < seqLen; i++) {
                    vpids[i] = promptLen - seqLen + i;
                    for (int j = i + 1; j < seqLen; j++) {
                        vmask[i * promptLen + (promptLen - seqLen + j)] = 1;
                    }
                }
                attentionMask.CopyFrom(Data(DataType::FLOAT32, {seqLen, promptLen}, vmask));
            } else {
                attentionMask = Data();
            }
        } else {
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, inputTokens[0]));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) promptLen + index - 1}));
        }
    }

    // 根据输入的tokens生成LLM推理的输入
    void basellm::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                     const std::vector<std::map<std::string, int>> &params, fastllm::Data &inputIds,
                                     fastllm::Data &attentionMask, fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
                    vpids[i * maxLen + base + j] = j;
                }

                std::fill(vmask.data() + i * maxLen * maxLen,
                        vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                            vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    void basellm::SetAdapter(const std::string &name) {
        if (weight.peftDict.find(name) == weight.peftDict.end()) {
            ErrorInFastLLM("Can`t find adapter name: " + name);
        }
        adapterName = name;
    }

    void basellm::DisableAdapter() {
        adapterName = "";
    }

    bool basellm::SetSaveHistoryChat(bool save) {
        if (this->model_type == "llama" || 
            this->model_type == "moe" || 
            this->model_type == "internlm" || 
            this->model_type == "qwen2_moe" || 
            this->model_type == "deepseek_v2" ||
            this->model_type == "qwen") {
            this->saveHistoryChat = save;
            return true;
        }
        return false;
    }

    void basellm::SetDataType(DataType dataType) {
        if (dataType == DataType::FLOAT32) {

        } else if (dataType == DataType::FLOAT16) {
            AssertInFastLLM(this->model_struct == "chatglm" || 
                            this->model_struct == "llama" ||
                            this->model_struct == "graph", 
                            this->model_struct + " doesn't support float16");
        } else {
            ErrorInFastLLM("SetDataType Error: datatype should be float32 or float16");
        }
        this->dataType = dataType;
    }

    JinjaVar ChatMessagesToJinjaVar(const ChatMessages &messages) {
        JinjaVar ret = {{"messages", fastllm::JinjaArray {}}};
        for (auto &message : messages) {
            ret["messages"].arrayValue.push_back({
                {"role", message.first},
                {"content", message.second}
            });
        }
        ret["add_generation_prompt"] = fastllm::JinjaVar{1};
        return ret;
    }

    std::string basellm::ApplyChatTemplate(const ChatMessages &messages) {
        if (this->weight.tokenizer.chatTemplate == "") {
            std::string ret = "";
            std::string user = "";
            int round = 0;
            for (auto &message : messages) {
                if (message.first == "user") {
                    user = message.second;
                } else if (message.first == "assistant") {
                    ret = MakeHistory(ret, round++, user, message.second);
                }
            }
            ret = MakeInput(ret, round, user);
            return ret;
        }
        return ApplyChatTemplate(ChatMessagesToJinjaVar(messages));
    }

    std::vector <int> basellm::ApplyChatTemplateToTokens(const ChatMessages &messages) {
        auto prompt = this->ApplyChatTemplate(messages);
        auto input = this->weight.tokenizer.Encode(prompt);
        std::vector<int> tokens;
        for (int i = 0; i < input.Count(0); i++) {
            tokens.push_back(((float *) input.cpuData)[i]);
        }
        return tokens;
    }

    std::string basellm::ApplyChatTemplate(const JinjaVar &var) {
        AssertInFastLLM(this->weight.tokenizer.chatTemplate != "", 
                        "ApplyChatTemplate error: model doesn't has chat_template.");
        JinjaVar local = var;
        for (auto &it : this->weight.tokenizer.tokenizerConfig.object_items()) {
            if (it.first != "messages" && it.second.is_string()) {
                local[it.first] = it.second.string_value();
            } else if (it.first.find_last_of("_token") != std::string::npos && it.second.is_object()) {
                local[it.first] = it.second["content"].string_value();
            }
        }
        JinjaTemplate temp = JinjaTemplate(this->weight.tokenizer.chatTemplate);
        return temp.Apply(local);
    }

    std::vector <int> basellm::ApplyChatTemplateToTokens(const JinjaVar &var) {
        auto prompt = this->ApplyChatTemplate(var);
        auto input = this->weight.tokenizer.Encode(prompt);
        std::vector<int> tokens;
        for (int i = 0; i < input.Count(0); i++) {
            tokens.push_back(((float *) input.cpuData)[i]);
        }
        return tokens;    
    }

    

    SLICE::SLICE(SLICEScheduler* scheduler) : scheduler(scheduler) {
        
    }

    SLICE::~SLICE() {
      
    }

    void SLICE::FindJob(double ratios) {
        std::lock_guard<std::mutex> lock(SLICEMutex);

        // 清空当前任务列表
        taskList.clear();

        // 预设 l(1) 到 l(16) 的值，这里可以根据实际需求调整
        std::vector<int> l_values = {14, 27, 38, 50, 60, 72, 83, 125,126,123,124,125,126,123,124}; 

        // 收集所有有效任务并按1000/TPOT降序排序
        std::vector<std::pair<int, double>> sortedTasks; // {handleId, 1000/TPOT}

        for (int priority = 0; priority < 4; priority++) {
            if (scheduler->priorityQueues[priority].empty()) {
                continue;
            }
            
            for (int handleId : scheduler->priorityQueues[priority]) {
                // 检查任务是否仍然有效
                if (scheduler->handleToContext.find(handleId) == scheduler->handleToContext.end() ||
                    scheduler->handleToContext[handleId]->isEnding) {
                    continue;
                }
                
                double tpot_ratio = 1000.0 / scheduler->handleToContext[handleId]->TPOT;
                sortedTasks.push_back({handleId, tpot_ratio});
            }
        }

        double period = 1000.0 * ratios;

        // 迭代选择任务
        for (int n = 1; n <= std::min(16, (int)sortedTasks.size()); n++) {
            // 计算 f(n)
            double f_n = 0.0;
            
            // 累加公式: f(n) = (1000/Job_n->TPOT) * l(n) + (1000/Job_{n-1}->TPOT - 1000/Job_n->TPOT) * l(n-1) + ...
            for (int i = 1; i <= n; i++) {
                if (i <= sortedTasks.size()) {
                    double tpot_i = sortedTasks[i-1].second; // 1000/TPOT_i
                    double tpot_prev = (i > 1) ? sortedTasks[i-2].second : 0.0; // 1000/TPOT_{i-1}
                    
                    if (i == n) {
                        // 最后一项: (1000/Job_n->TPOT) * l(n)
                        f_n += tpot_i * l_values[i-1];
                    } else {
                        // 中间项: (1000/Job_{i-1}->TPOT - 1000/Job_i->TPOT) * l(i)
                        f_n += (tpot_prev - tpot_i) * l_values[i-1];
                    }
                }
            }


            
            // 检查 f(n) <= period
            if (f_n <= period) {
                // 加入第n个任务到taskList
                int handleId = sortedTasks[n-1].first;
                int fillCount = (int)sortedTasks[n-1].second; // 1000/TPOT 转换为整数
                taskList.push_back({handleId, fillCount});
                
                // 继续查看下一个任务
                continue;
            } else {
                // f(n) > (1000.0 * ratios)，不满足条件，停止添加
                break;
            }
        }

        // 更新总token数
        selecttotalToken = 0;
        for (const auto& task : taskList) {
            selecttotalToken += task.second;
        }




    }

    void SLICE::ConstructMatrix() {
        std::lock_guard<std::mutex> lock(SLICEMutex);


        dynamicMatrix.clear();
        dynamicMatrixSize = 0;
        dynamicMatrixIndex = 1;

        // 按1000/TPOT降序排序
        std::sort(taskList.begin(), taskList.end(), 
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second > b.second;
        });

        // 构建矩阵，按降序排列的行任务
        dynamicMatrixSize = taskList.size();
        for(int i = 0; i < dynamicMatrixSize; i++) {
            int taskHandleId = taskList[i].first;
            int fillCount = taskList[i].second;
            
            // 创建新的矩阵行
            dynamicMatrix.push_back(std::array<int, 50>());
            dynamicMatrix[i][0] = taskHandleId; // 第一列存储任务ID
            
            // 根据fillCount填入1，其余填0
            for(int j = 1; j < 50; j++) {
                if (j <= fillCount) {
                    dynamicMatrix[i][j] = 1;
                } else {
                    dynamicMatrix[i][j] = 0;
                }
            }
        }
    }

    void SLICE::SelectMatrixJob() {
        std::lock_guard<std::mutex> lock(SLICEMutex);
        
        //取得矩阵第一个任务需要生成最大token
        int temp = dynamicMatrix[0][0];
        MatrixTokenMax = (1000/scheduler->handleToContext[temp]->TPOT) + 1;

        //根据矩阵选择任务
        for(int i = 0; i < dynamicMatrixSize; i++) {
            // 检查当前列（dynamicMatrixIndex）是否为1
            if(dynamicMatrix[i][dynamicMatrixIndex] == 1) {
                int handleId = dynamicMatrix[i][0]; 
                // 加入任务
                scheduler->currentBatch.insert(handleId);
            } else if(dynamicMatrixIndex > MatrixTokenMax || Callmatrix == true) {
                dynamicMatrixIndex = 0;
                break;
            }
        }

    }

    void SLICE::RemoveMatrixJob(int handleId) {
        std::lock_guard<std::mutex> lock(SLICEMutex);

        //在矩阵中移除该任务
        for (auto it = dynamicMatrix.begin(); it != dynamicMatrix.end(); ) {
            if ((*it)[0] == handleId) {  // 第一列存储任务ID
                it = dynamicMatrix.erase(it);  // 移除这一行
                dynamicMatrixSize--;  // 更新矩阵大小
                break;
            } else {
                ++it;
            }
        }

    }





    

    SLICEScheduler::SLICEScheduler() : slice(this) { 
        
        // 确保优先级队列初始化为空
        for (int i = 0; i < 4; i++) {
            priorityQueues[i].clear();
        }
        // 确保映射初始化为空
        handleToPriority.clear();
        currentBatch.clear();
        // 确保historicalLengths为空
        historicalLengths.clear();
        predictedAvgLength = 128.0f;
        lmsAlpha = 0.1f;
    }
    
    void SLICEScheduler::AddTask(int handleId, ResponseContext* context, bool isRealtime, bool isShortTask, int priority, int reward, int TPOT, int u) {
        if (!context) {
            printf("Warning: Trying to add a null context to SLICEScheduler\n");
            return;
        }
        std::lock_guard<std::mutex> lock(schedulerMutex);
        // 设置任务属性
        context->isRealtime = isRealtime;
        context->isShortTask = isShortTask;
        reward = u * TPOT;
        context->reward = reward;
        
        context->TPOT = TPOT;
        // 确定优先级
        context->priority = priority;
        handleToPriority[handleId] = priority;
        // 添加到相应的优先级队列 按reward值降序排列
        int n = priorityQueues[priority].size();
        bool inserted = false;
        
        for (int i = 0; i < n; i++) {
            if (handleToContext[priorityQueues[priority][i]]->reward < context->reward) {
                priorityQueues[priority].insert(priorityQueues[priority].begin() + i, handleId);
                inserted = true;
                break;
            }
        }
        
        // 如果没有找到插入位置（reward最大）或队列为空，添加到末尾
        if (!inserted) {
            priorityQueues[priority].push_back(handleId);
        }

        //赋值任务到映射
        handleToContext[handleId] = context;
        // 记录任务开始时间
        handleStartTime[handleId] = std::chrono::system_clock::now();

        slice.Callmatrix = true;
 
    }


    
    int SLICEScheduler::GetNextBatch(int tokencountindex, double ratios) {

         std::lock_guard<std::mutex> lock(schedulerMutex);

        if(slice.Callmatrix == true) {
        
        slice.FindJob(ratios);

        slice.ConstructMatrix();

        slice.Callmatrix = false;

        }
        
        // 每次调用选择新一轮任务，先清空当前批次
        if (slice.dynamicMatrixSize > 0 && slice.Callmatrix == false) {
            currentBatch.clear();
            slice.SelectMatrixJob();
            slice.dynamicMatrixIndex++;
        }
   
        // 返回本次选择的总token预算
        return slice.selecttotalToken;
    }
    
    void SLICEScheduler::TaskCompleted(int handleId) {
        std::lock_guard<std::mutex> lock(schedulerMutex);

        // 任务完成，需要重新构建矩阵
        slice.Callmatrix = true;

        // 从当前批处理中移除
        currentBatch.erase(handleId);
        int temp = handleToPriority[handleId];

        // 从优先级映射中移除
        handleToPriority.erase(handleId);
        // 记录并打印任务完成时间和token数
        auto endTime = std::chrono::system_clock::now();
        auto contextIt = handleToContext.find(handleId);
        if (handleStartTime.find(handleId) != handleStartTime.end()) {
            if (contextIt != handleToContext.end() && contextIt->second) {
            auto startTime = handleStartTime[handleId];
            float duration = fastllm::GetSpan(startTime, endTime);
            int TPOT = contextIt->second->TPOT; 
            int priority = contextIt->second->priority;
            int tokens = contextIt->second->curTokens;
            EGstats.addResult(priority, duration, tokens);

            // 输出token数需要从context获取
            handleStartTime.erase(handleId);
            handleToContext.erase(contextIt);
            }
        } else {
            printf("任务编号: %d 完成, 未找到开始时间记录\n", handleId);
        }
        completedTasks++;
        if (completedTasks == totalTasks && totalTasks > 0) {
        EGstats.printStats();
        }

        // 在任务结束后，从优先级队列中移除这些任务
        // 找到任务所在的优先级队列并移除
        for (int priority = 0; priority < 4; priority++) {
            auto& queue = priorityQueues[priority];
            auto it = std::find(queue.begin(), queue.end(), handleId);
            if (it != queue.end()) {
                queue.erase(it);
                std::cout << "任务开始处理时从优先级 " << priority << " 队列中移除任务: " << handleId << std::endl;
                break; // 任务只会在一个队列中
            }
        }
        
    }

    
    void SLICEScheduler::UpdateTaskUtility(int handleId, int newUtility) {
         std::lock_guard<std::mutex> lock(schedulerMutex);


    }



}

