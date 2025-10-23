//
// Created by huangyuyang on 6/27/23.
//

#include "model.h"

#include <cstring>

#ifdef WIN32
#define DLL_EXPORT _declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {
    DLL_EXPORT void print_cpu_ins() {
        fastllm::PrintInstructionInfo();
    }

    DLL_EXPORT void set_cpu_threads(int threads) {
        fastllm::SetThreads(threads);
    }

    DLL_EXPORT int get_cpu_threads() {
        return fastllm::GetThreads();
    }

    DLL_EXPORT void set_cpu_low_mem(bool low) {
        fastllm::SetLowMemMode(low);
    }

    DLL_EXPORT void set_cuda_embedding(bool cuda_embedding) {
        fastllm::SetCudaEmbedding(cuda_embedding);
    }

    DLL_EXPORT bool get_cpu_low_mem(bool low) {
        return fastllm::GetLowMemMode();
    }

    DLL_EXPORT void set_kvcache_in_cpu(bool in) {
        fastllm::SetKVCacheInCPU(in);
    }

    DLL_EXPORT bool get_kvcache_in_cpu() {
        return fastllm::GetKVCacheInCPU();
    }

    DLL_EXPORT void set_device_map(int device_cnt, int *lens, char *devices, int *values) {
        std::map <std::string, int> deviceMap;
        int cur = 0;
        for (int i = 0; i < device_cnt; i++) {
            std::string key = "";
            for (int j = 0; j < lens[i]; j++) {
                key += devices[cur++];
            }
            deviceMap[key] = values[i];
        }
        fastllm::SetDeviceMap(deviceMap);
    }

    DLL_EXPORT struct ModelManager {
        std::mutex locker;
        std::map <int, std::unique_ptr<fastllm::basellm> > models;

        fastllm::basellm *GetModel(int handle) {
            locker.lock();
            auto ret = models[handle].get();
            locker.unlock();
            return ret;
        }
    };

    static ModelManager models;

    DLL_EXPORT char *string_to_chars(const std::string &s) {
        char *svalue = new char[s.size() + 1];
        memcpy(svalue, s.data(), s.size());
        svalue[s.size()] = 0;
        return svalue;
    }

    DLL_EXPORT fastllm::GenerationConfig make_config(int max_length, bool do_sample, float top_p, int top_k,
                                          float temperature, float repeat_penalty, bool output_logits, bool add_special_tokens) {
        fastllm::GenerationConfig config;
        config.output_token_limit = max_length;
        config.temperature = temperature;
        config.repeat_penalty = repeat_penalty;
        if (do_sample) {
            config.top_p = top_p;
            config.top_k = top_k;
        }
        config.output_logits = output_logits;
        config.add_special_tokens = add_special_tokens;
        return config;
    }

    DLL_EXPORT int create_llm_model(char *path) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateLLMModelFromFile(path);
        models.locker.unlock();
        return id;
    }

    DLL_EXPORT int create_llm_model_fromhf(char *path, int dataType, int groupCnt, bool skipTokenizer) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateLLMModelFromHF(path, (fastllm::DataType)dataType, groupCnt, skipTokenizer);
        models.locker.unlock();
        return id;
    }

    DLL_EXPORT int create_llm_model_fromhf_with_config(char *path, int dataType, int groupCnt, bool skipTokenizer, char *config) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateLLMModelFromHF(path, (fastllm::DataType)dataType, groupCnt, skipTokenizer, (std::string)config);
        models.locker.unlock();
        return id;
    }

    DLL_EXPORT int create_llm_tokenizer_fromhf(char *path) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateLLMTokenizerFromHF(path);
        models.locker.unlock();
        return id;
    }

    DLL_EXPORT int create_empty_llm_model(char *type) {
        models.locker.lock();
        int id = models.models.size();
        models.models[id] = fastllm::CreateEmptyLLMModel(type);
        models.locker.unlock();
        return id;
    }

    DLL_EXPORT int get_tokenizer_vocab_size(int modelId) {
        auto model = models.GetModel(modelId);
        int ret = model->weight.tokenizer.tokenToStringDict.size();
        return ret;
    }

    DLL_EXPORT void add_tokenizer_word_llm_model(int modelId, char *key, int tokenId, float score) {
        auto model = models.GetModel(modelId);
        model->weight.AddTokenizerWord(key, tokenId, score);
        return;
    }

    DLL_EXPORT void set_special_tokens_llm_model(int modelId, int token_cnt, int *lens, char *tokens, int *ids) {
        std::map <std::string, int> tokenMap;
        int cur = 0;
        for (int i = 0; i < token_cnt; i++) {
            std::string key = "";
            for (int j = 0; j < lens[i]; j++) {
                key += tokens[cur++];
            }
            tokenMap[key] = ids[i];
        }
        auto model = models.GetModel(modelId);
        model->weight.tokenizer.SetSpecialTokens(tokenMap);
        return;
    }

    DLL_EXPORT int token_decode(int modelId, int tokenId, int output_buffer_len, char *output_buffer) {
        // 正常时候返回0，输出buffer长度不足时返回输出的bytes数量，包含末尾的\0
        if(tokenId == -1) {
            output_buffer[0] = '\0';
            return 0;
        }
        auto model = models.GetModel(modelId);
        std::string s = model->weight.tokenizer.DecodeTokens(std::vector <int> {tokenId});
        if(s.length() + 1 > output_buffer_len) {
            return (int)s.length() + 1;
        }
        memcpy(output_buffer, s.c_str(), s.length() + 1);
        return 0;
    }

    DLL_EXPORT int token_encode_string(int modelId, char *content, int output_buffer_len, int *output_buffer) {
        // 返回写入到output_buffer中的数量。当output不足时候，只输出对应的部分
        auto model = models.GetModel(modelId);
        auto v = model->weight.tokenizer.Encode(content);
        for (int i = 0; i < v.Count(0); i++) {
            if(i >= output_buffer_len) {
                break;
            }
            output_buffer[i] = (int)((float*)v.cpuData)[i];
        }
        return (int)v.Count(0);
    }

    DLL_EXPORT void add_dict_llm_model(int modelId, char *key, char *value) {
        auto model = models.GetModel(modelId);
        model->weight.AddDict(key, value);
        return;
    }

    DLL_EXPORT void add_adapter_dict_llm_model(int modelId, char *adapterName, char *key, char *value) {
        auto model = models.GetModel(modelId);
        model->weight.AddAdapterDict(adapterName, key, value);
        return;
    }

    DLL_EXPORT void set_adapter(int modelId, char *name) {
        auto model = models.GetModel(modelId);
        model->SetAdapter(name);
        return;
    }

    DLL_EXPORT void disable_adapter(int modelId, char *name) {
        auto model = models.GetModel(modelId);
        model->DisableAdapter();
        return;
    }

    DLL_EXPORT void release_memory(int modelId) {
        auto model = models.GetModel(modelId);
        model->weight.ReleaseWeight();
        return;
    }

    DLL_EXPORT void set_save_history(int modelId, bool save) {
        auto model = models.GetModel(modelId);
        model->SetSaveHistoryChat(save);
        return;
    }

    DLL_EXPORT void set_model_atype(int modelId, char *atype) {
        auto model = models.GetModel(modelId);
        std::string atypeStr = atype;
        if (atypeStr == "float16" || atypeStr == "half") {
            model->SetDataType(fastllm::DataType::FLOAT16);
        } else if (atypeStr == "float" || atypeStr == "float32") {
            model->SetDataType(fastllm::DataType::FLOAT32);
        } else {
            fastllm::ErrorInFastLLM("set_model_atype error: atype should be float32 or float16.");
        }
        return;
    }

    DLL_EXPORT void init_params_llm_model(int modelId) {
        auto model = models.GetModel(modelId);
        model->InitParams();
        return;
    }

    DLL_EXPORT void warmup_llm_model(int modelId) {
        auto model = models.GetModel(modelId);
        model->WarmUp();
        return;
    }

    DLL_EXPORT void save_llm_model(int modelId, char *path) {
        auto model = models.GetModel(modelId);
        model->SaveModel(path);
        return;
    }

    DLL_EXPORT void add_weight_llm_model(int modelId, char *key, int dimsLen, void *dimsData,
                              int dataType, int weightType, int oriDataType, void *oriData, int groupCnt) {
        auto model = models.GetModel(modelId);
        std::vector <int> dims = std::vector <int> (dimsLen);
        for (int i = 0; i < dims.size(); i++) {
            dims[i] = ((int*)dimsData)[i];
        }
        model->weight.AddWeight(key, dims,
                                (fastllm::DataType)dataType,
                                (fastllm::WeightType)weightType,
                                (fastllm::DataType)oriDataType,
                                (uint8_t*)oriData, groupCnt);
        return;
    }

    DLL_EXPORT void add_qlinear_weight_llm_model(int modelId, char *key, int dimsLen, void *dimsData,
                                                 int bit, void *scales, void *oriData) {
        auto model = models.GetModel(modelId);
        std::vector <int> dims = std::vector <int> (dimsLen);
        for (int i = 0; i < dims.size(); i++) {
            dims[i] = ((int*)dimsData)[i];
        }
        model->weight.AddQLinearWeight(key, dims, bit, (float*)scales, (uint8_t*)oriData);
        return;
    }

    DLL_EXPORT char *make_input_llm_model(int modelId, char *history, int round, char *input) {
        auto model = models.GetModel(modelId);
        char *ret = string_to_chars(model->MakeInput(history, round, input));
        return ret;
    }

    DLL_EXPORT char *make_history_llm_model(int modelId, char *history, int round, char *input, char *output) {
        auto model = models.GetModel(modelId);
        return string_to_chars(model->MakeHistory(history, round, input, output));
    }

    DLL_EXPORT char *apply_chat_template(int modelId, char *str, int cnt, int *pos, int *len) {
        auto model = models.GetModel(modelId);
        fastllm::ChatMessages messages;
        for (int i = 0; i < cnt / 2; i++) {
            std::string role, content;
            for (int j = 0; j < len[i * 2]; j++) {
                role += str[pos[i * 2] + j];
            }
            for (int j = 0; j < len[i * 2 + 1]; j++) {
                content += str[pos[i * 2 + 1] + j];
            }
            messages.push_back(std::make_pair(role, content));
        }
        return string_to_chars(model->ApplyChatTemplate(messages));
    }

    DLL_EXPORT void add_eos_token(int modelId, char *str, int len) {
        std::string eos_token = "";
        for (int i = 0; i < len; i++) {
            eos_token += str[i];
        }
        auto model = models.GetModel(modelId);
        model->eos_token_id = model->weight.tokenizer.GetTokenId(eos_token);
        model->eos_token_ids.insert(model->weight.tokenizer.GetTokenId(eos_token));
    }

    DLL_EXPORT char *response_str_llm_model(int modelId, char *content,
                                 int max_length, bool do_sample, float top_p, int top_k,
                                 float temperature, float repeat_penalty, bool output_logits) {
        auto model = models.GetModel(modelId);
        auto config = make_config(max_length, do_sample, top_p, top_k, temperature, repeat_penalty, output_logits, true);
        std::string s = model->Response(content, nullptr, config);
        return string_to_chars(s);
    }

    DLL_EXPORT int launch_response_str_llm_model(int modelId, char *content,
                                      int max_length, bool do_sample, float top_p, int top_k,
                                      float temperature, float repeat_penalty, bool output_logits,
                                      int stop_token_len, int * stop_token_ids) {
        auto model = models.GetModel(modelId);
        std::vector <int> tokens;
        auto v = model->weight.tokenizer.Encode(content);
        for (int i = 0; i < v.Count(0); i++) {
            tokens.push_back((int)((float*)v.cpuData)[i]);
        }
        auto config = make_config(max_length, do_sample, top_p, top_k, temperature, repeat_penalty, output_logits, true);
        for(int i = 0; i < stop_token_len; i++ )
        {
            config.stop_token_ids.insert(stop_token_ids[i]);
        }
        return model->LaunchResponseTokens(tokens, config);
    }

    // 尝试fetch，如果能fetch成功则返回true（后续需要fetch一下)，用于异步操作
    DLL_EXPORT bool can_fetch_response_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        return model->CanFetchResponse(handleId);
    }

    // 终止handleId的请求
    DLL_EXPORT void abort_response_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        model->AbortResponse(handleId);
    }

    DLL_EXPORT char *fetch_response_str_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        int ret = model->FetchResponseTokens(handleId);
        std::string s = (ret == -1 ? "<flmeos>" : model->weight.tokenizer.DecodeTokens(std::vector <int> {ret}));
        return string_to_chars(s);
    }

    DLL_EXPORT int launch_response_llm_model(int modelId, int len, int *values,
                                  int max_length, bool do_sample, float top_p, int top_k,
                                  float temperature, float repeat_penalty, bool output_logits,
                                  int stop_token_len, int * stop_token_ids) {
        std::vector <int> input;
        for (int i = 0; i < len; i++) {
            input.push_back(values[i]);
        }
        auto config = make_config(max_length, do_sample, top_p, top_k, temperature, repeat_penalty, output_logits, false);
        for(int i = 0; i < stop_token_len; i++ )
        {
            config.stop_token_ids.insert(stop_token_ids[i]);
        }
        auto model = models.GetModel(modelId);
        return model->LaunchResponseTokens(input, config);
    }

    DLL_EXPORT int fetch_response_llm_model(int modelId, int handleId) {
        auto model = models.GetModel(modelId);
        return model->FetchResponseTokens(handleId);
    }

    DLL_EXPORT int fetch_response_logits_llm_model(int modelId, int handleId, float *logits) {
        auto model = models.GetModel(modelId);
        std::vector <float> retLogits;
        int ret = model->FetchResponseLogits(handleId, retLogits);
        if (ret != -1) {
            memcpy(logits, retLogits.data(), retLogits.size() * sizeof(float));
        }
        return ret;
    }

    DLL_EXPORT void add_cache_llm_model(int modelId, int len, int *values) {
        std::vector <int> input;
        for (int i = 0; i < len; i++) {
            input.push_back(values[i]);
        }
        auto model = models.GetModel(modelId);
        model->AddPromptCache(input);
    }

    DLL_EXPORT void set_kv_cache_limit_llm_model(int modelId, long long bytes) {
        auto model = models.GetModel(modelId);
        model->kvCacheLimit = bytes;
    }

    DLL_EXPORT void set_max_batch_llm_model(int modelId, int batch) {
        auto model = models.GetModel(modelId);
        model->maxBatch = batch;
    }

    DLL_EXPORT void set_verbose_llm_model(int modelId, bool verbose) {
        auto model = models.GetModel(modelId);
        model->verbose = verbose;
    }

    DLL_EXPORT int get_max_input_len_llm_model(int modelId) {
        auto model = models.GetModel(modelId);
        return model->max_positions;
    }
};
