#include "utils.h"
#include "json11.hpp"

#include "model.h"
#include "fastllm.h"
#include <sstream>
#include <fstream>

#include "chatglm.h"
#include "moss.h"
#include "llama.h"
#include "moe.h"
#include "deepseekv2.h"
#include "qwen.h"
#include "glm.h"
#include "minicpm.h"
#include "internlm2.h"
#include "bert.h"
#include "graphllm.h"

namespace fastllm {
    std::string ReadAllFile(const std::string &fileName) {
        std::ifstream t(fileName.c_str(), std::ios::in);
        if (!t.good()) {
            ErrorInFastLLM("Read error: can't find \"" + fileName + "\".");
        }

        std::string ret((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        t.close();
        return ret;
    }

    void ConvertDataType(uint8_t *src, DataType srcDtype, uint8_t *dst, DataType dstDtype, uint64_t len) {
        if (srcDtype == dstDtype) {
            int unitSize = 4;
            if (dstDtype == DataType::FLOAT32) {
                unitSize = 4;
            } else if (dstDtype == DataType::FLOAT16 || dstDtype == DataType::BFLOAT16) {
                unitSize = 2;
            } else {
                ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");    
            }
            memcpy(dst, src, len * unitSize);
        } else if (srcDtype == DataType::BFLOAT16 && dstDtype == DataType::FLOAT32) {
            uint16_t *u16dst = (uint16_t*)dst;
            uint16_t *u16src = (uint16_t*)src;
            for (size_t i = 0; i < len; i++) {
                u16dst[i * 2] = 0;
                u16dst[i * 2 + 1] = u16src[i];
            }
        } else if (srcDtype == DataType::FLOAT16 && dstDtype == DataType::FLOAT32) {
            float *fdst = (float*)dst;
            uint16_t *u16src = (uint16_t*)src;
            for (size_t i = 0; i < len; i++) {
                fdst[i] = half_to_float(u16src[i]);
            }
        } else {
            ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
        }
    }

    void basellm::LoadFromFile(const std::string &fileName) {
        this->weight.LoadFromFile(fileName);
        this->InitParams();
    }

    void basellm::InitParams() {
        if (this->weight.dicts.find("bos_token_id") != this->weight.dicts.end()) {
            if (this->weight.dicts["bos_token_id"]!="None") {
                this->bos_token_id = atoi(this->weight.dicts["bos_token_id"].c_str());
            }
        }
        if (this->weight.dicts.find("eos_token_id") != this->weight.dicts.end()) {
            if (this->weight.dicts["eos_token_id"]!="None") {
                if (this->weight.dicts["eos_token_id"][0] == '[' && this->eos_token_ids.empty()) {
                    std::string error;
                    json11::Json ids = json11::Json::parse(this->weight.dicts["eos_token_id"], error);
                    for (auto &it : ids.array_items()) {
                        this->eos_token_ids.insert(it.int_value());
                    }
                } else {
                    this->eos_token_id = atoi(this->weight.dicts["eos_token_id"].c_str());
                }
            }
        }
        if (this->weight.dicts.find("im_start_id") != this->weight.dicts.end()) {
            this->bos_token_id = atoi(this->weight.dicts["im_start_id"].c_str());
            this->eos_token_id = atoi(this->weight.dicts["im_end_id"].c_str());
        }
        if (this->weight.dicts.find("num_hidden_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
        }else if (this->weight.dicts.find("num_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_layers"].c_str());
        }
        if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
            embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
        }
        if (this->weight.dicts.find("num_attention_heads") != this->weight.dicts.end()) {
            num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
        }
        if (this->weight.dicts.find("pre_prompt") != this->weight.dicts.end()) {
            pre_prompt = this->weight.dicts["pre_prompt"];
        }
        if (this->weight.dicts.find("user_role") != this->weight.dicts.end()) {
            user_role = this->weight.dicts["user_role"];
        }
        if (this->weight.dicts.find("bot_role") != this->weight.dicts.end()) {
            bot_role = this->weight.dicts["bot_role"];
        }
        if (this->weight.dicts.find("history_sep") != this->weight.dicts.end()) {
            history_sep = this->weight.dicts["history_sep"];
        }
        if (this->weight.dicts.find("tokenizer_add_dummy_prefix") != this->weight.dicts.end()) {
            std::string value = this->weight.dicts["tokenizer_add_dummy_prefix"];
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            std::istringstream iss(value);
            iss >> std::boolalpha >> this->weight.tokenizer.addDummyPrefix;
        }
        if (this->weight.dicts.find("tokenizer_remove_extra_whitespaces") != this->weight.dicts.end()) {
            std::string value = this->weight.dicts["tokenizer_remove_extra_whitespaces"];
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            std::istringstream iss(value);
            iss >> std::boolalpha >> this->weight.tokenizer.removeExtraWhitespaces;
        }
        if (this->weight.dicts.find("tokenizer_byte_as_char") != this->weight.dicts.end()) {
            std::string value = this->weight.dicts["tokenizer_byte_as_char"];
            transform(value.begin(), value.end(), value.begin(), ::tolower);
            std::istringstream iss(value);
            iss >> std::boolalpha >> this->weight.tokenizer.byteAsChar;
        }

        this->deviceMap = GetDeviceMap();
    }

    void basellm::SaveLowBitModel(const std::string &fileName, int bit) {
        this->weight.SaveLowBitModel(fileName, bit);
    }

    void basellm::SaveModel(const std::string &fileName) {
        if (this->weight.tokenizer.chatTemplate.empty()) {
            if (this->weight.dicts.find("pre_prompt") == this->weight.dicts.end())
                this->weight.dicts["pre_prompt"] = pre_prompt;
            if (this->weight.dicts.find("user_role") == this->weight.dicts.end())
                this->weight.dicts["user_role"] = user_role;
            if (this->weight.dicts.find("bot_role") == this->weight.dicts.end())
                this->weight.dicts["bot_role"] = bot_role;
            if (this->weight.dicts.find("history_sep") == this->weight.dicts.end())
                this->weight.dicts["history_sep"] = history_sep;
        }
        this->weight.SaveLowBitModel(fileName, 0);
    }

    fastllm::basellm *CreateModelWithType(const std::string &modelType) {
        basellm *model = nullptr;
        if (modelType == "chatglm") {
            model = (basellm*)(new ChatGLMModel());
        } else if (modelType == "moss") {
            model = (basellm*)(new MOSSModel());
            model->weight.tokenizer.type = Tokenizer::TokenizerType::BPE;
            model->eos_token_id = 106068;
        } else if (modelType == "baichuan") {
            model = (basellm*)(new LlamaModel());
            model->model_type = "baichuan";
            model->pre_prompt = "";
            model->user_role = "<human>:";
            model->bot_role = "\n<bot>:";
            model->history_sep = "\n";
            model->weight.tokenizer.type = Tokenizer::TokenizerType::BPE;
        } else if (modelType == "internlm") {
            model = new LlamaModel();
            model->model_type = "internlm";
        } else if (modelType == "internlm2") {
            model = new Internlm2Model();
            model->model_type = "internlm";
        } else if (modelType == "llama") {
            model = (basellm*)(new LlamaModel());
        } else if (modelType == "moe" || modelType == "qwen2_moe") {
            model = (basellm*)(new MoeModel());
        } else if (modelType == "deepseek_v2") {
            model = (basellm*)(new DeepSeekV2Model());
        } else if (modelType == "qwen2") {
            model = new LlamaModel();
            model->model_type = "qwen";
        } else if (modelType=="minicpm") {
            model = new MiniCpmModel();
        } else if (modelType == "qwen") {
            model = (basellm *) (new QWenModel());
            model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
        } else if (modelType == "glm") {
            model = (basellm*)(new GLMModel());
        } else if (modelType == "bert") {
            model = (basellm*)(new BertModel());
        } else if (modelType == "fastllmJson") {
            model = new GraphLLMModel("fastllmJson");
        } else {
            model = new GraphLLMModel(modelType);
        }
        return model;
    }

    std::unique_ptr<BertModel> CreateEmbeddingModelFromFile(const std::string &fileName) {
        BertModel *model = new BertModel();
        model->weight.tokenizer.type = Tokenizer::BERT;
        model->LoadFromFile(fileName);
        model->WarmUp();
        return std::unique_ptr<fastllm::BertModel> (model);
    }

    std::unique_ptr<fastllm::basellm> CreateLLMModelFromFile(const std::string &fileName) {
        std::string modelType = GetModelTypeFromFile(fileName);
        basellm *model = CreateModelWithType(modelType);
        model->LoadFromFile(fileName);
        model->WarmUp();
        return std::unique_ptr<fastllm::basellm> (model);
    }

    std::unique_ptr<basellm> CreateEmptyLLMModel(const std::string &modelType) {
        basellm *model = CreateModelWithType(modelType);
        return std::unique_ptr<fastllm::basellm> (model);
    }

    template <typename T>
    void TransposeSimple(T *pDst, T *pSrc, int dstStride, int srcStride, int n, int m) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }
    }
    extern void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m);

    struct SafeTensorItem {
        std::string tensorName;
        std::string fileName;
        std::string dtype;
        std::vector <std::uint64_t> shape;
        std::vector <int> intShape;
        std::vector <std::uint64_t> data_offsets;

        uint64_t len, bytes;
        uint8_t *buffer = nullptr;

        SafeTensorItem() {} 

        SafeTensorItem(const std::string &tensorName, const std::string &fileName, const json11::Json &config, uint64_t baseOffset) {
            this->tensorName = tensorName;
            this->fileName = fileName;

            this->dtype = config["dtype"].string_value();
            for (auto &it : config["data_offsets"].array_items()) {
                this->data_offsets.push_back(baseOffset + it.ll_value());
            }
            for (auto &it : config["shape"].array_items()) {
                this->shape.push_back(it.ll_value());
                this->intShape.push_back(this->shape.back());
            }

            len = 1;
            for (auto &it : shape) {
                len *= it;
            }
            bytes = this->data_offsets[1] - this->data_offsets[0];
        }

        void CreateBuffer(DataType dstType) {
            DataType srcType;
            if (this->dtype == "BF16") {
                srcType = DataType::BFLOAT16;
            } else if (this->dtype == "F16") 
            {
                srcType = DataType::FLOAT16;
            } else {
                ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
            }
            
            int unitSize = 4;
            if (dstType == DataType::FLOAT32) {
                unitSize = 4;
            } else if (dstType == DataType::FLOAT16 || dstType == DataType::BFLOAT16) {
                unitSize = 2;
            } else {
                ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport dst dtype " + std::to_string(dstType) + "\n");
            }
            ClearBuffer();
            buffer = new uint8_t[(size_t)len * unitSize];

            FILE *fi = fopen(this->fileName.c_str(), "rb");
            int ret;
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(fi, this->data_offsets[0], 0);
#else
            fseek(fi, this->data_offsets[0], 0);
#endif
            if (dstType == srcType) {
                ret = fread(buffer, 1, this->bytes, fi);
            } else {
                uint8_t *ori = new uint8_t[this->bytes];
                ret = fread(ori, 1, this->bytes, fi);
                ConvertDataType(ori, srcType, buffer, dstType, len);
                delete[] ori;
            }
            fclose(fi);
        }

        void Transpose(DataType type) {
            int n = intShape[0], m = intShape[1];
            if (type == DataType::FLOAT32) {
                float *temp = new float[len];
                memcpy(temp, this->buffer, len * sizeof(float));
                fastllm::Transpose((float*)this->buffer, temp, n, m, n, m);
                delete[] temp;
            } else if (type == DataType::FLOAT16 || type == DataType::BFLOAT16) {
                uint16_t *temp = new uint16_t[len];
                memcpy(temp, this->buffer, len * sizeof(uint16_t));
                TransposeSimple((uint16_t*)this->buffer, temp, n, m, n, m);
                delete[] temp;
            } else {
                ErrorInFastLLM("SafeTensorItem.Transpose: unsupport dtype " + std::to_string(type) + "\n");
            }
        }

        void ClearBuffer() {
            delete[] buffer;
            buffer = nullptr;
        }
    };

    struct SafeTensors {
        std::set <std::string> fileNames;
        std::map <std::string, SafeTensorItem> itmeDict;

        SafeTensors (const std::set <std::string> &fileNames) {
            std::string error;
            this->fileNames = fileNames;
            for (auto &fileName : fileNames) {
                FILE *f = fopen(fileName.c_str(), "rb");
                uint64_t configBytes;
                int ret = fread(&configBytes, 8, 1, f);
                char *configString = new char[configBytes + 5];
                ret = fread(configString, 1, configBytes, f);
                configString[configBytes] = 0;
                auto config = json11::Json::parse(configString, error);
                for (auto it : config.object_items()) {
                    if (it.first != "__metadata__" ) {
                        itmeDict[it.first] = SafeTensorItem(it.first, fileName, it.second, 8 + configBytes);
                    }
                }

                delete[] configString;
            }
        }

        std::vector <std::string> GetSortedItemNames() {
            std::vector <std::pair <std::pair <std::string, uint64_t>, std::string> > v;
            for (auto &it : itmeDict) {
                if (it.second.intShape.size() > 0 && it.second.dtype != "BOOL") {
                    v.push_back(std::make_pair(std::make_pair(it.second.fileName, it.second.data_offsets[0]), it.first));
                }
            }
            std::sort(v.begin(), v.end());
            std::vector <std::string> ret;
            for (int i = 0; i < v.size(); i++) {
                ret.push_back(v[i].second);
            }
            return ret;
        }
    };

    std::string Base64Decode(const std::string &encoded) {
        static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";
        int in_len = encoded.size();
        int i = 0, j = 0, in_ = 0;
        char char_array_4[4], char_array_3[3];
        std::string ret = "";
        
        while (in_len-- && ( encoded[in_] != '=')) {
            char_array_4[i++] = encoded[in_]; in_++;
            if (i == 4) {
                for (i = 0; i < 4; i++)
                    char_array_4[i] = base64_chars.find(char_array_4[i]);
                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
                for (i = 0; (i < 3); i++)
                    ret.push_back(char_array_3[i]);
                i = 0;
            }
        }

        if (i) {
            for (j = i; j < 4; j++)
                char_array_4[j] = 0;

            for (j = 0; j < 4; j++)
                char_array_4[j] = base64_chars.find(char_array_4[j]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
        }

        return ret;
    }

    void SplitString(const std::string &str, const std::set <char> &chars, std::vector <std::string> &ret) {
        ret.clear();
        std::string now = "";
        for (int i = 0; i < str.size(); i++) {
            if (chars.find(str[i]) == chars.end()) {
                now += str[i];
            } else {
                if (now != "") {
                    ret.push_back(now);
                    now = "";
                }
            }
        }
        if (now != "") {
            ret.push_back(now);
        }
    }

    void DealLLMTokenizerFromHFToModel(const std::string &path, basellm *model) {
        std::string error;
        std::string tokenizerConfigFile = path + "tokenizer_config.json";
        if (!fastllm::FileExists(tokenizerConfigFile)) {
            return;
        }
        auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
        model->weight.tokenizer.SetTokenizerConfig(tokenizerConfig);
        std::string tokenizerClass = tokenizerConfig["tokenizer_class"].string_value();
        if (tokenizerClass == "PreTrainedTokenizerFast" || tokenizerClass == "Qwen2Tokenizer") {
        } else if (tokenizerClass == "ChatGLM4Tokenizer") {
            // 历史遗留问题
            model->bot_role = " ";
        }
    }

    void LoadLLMTokenizerFromHFToModel(const std::string &path, basellm *model) {
        std::string error;
        std::string tokenizerConfigFile = path + "tokenizer_config.json";
        auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
        model->weight.tokenizer.SetTokenizerConfig(tokenizerConfig);
        if (!model->weight.tokenizer.chatTemplate.empty() && model->weight.dicts.find("chat_template") == model->weight.dicts.end())
            model->weight.AddDict("chat_template", model->weight.tokenizer.chatTemplate);
        std::string tokenizerClass = tokenizerConfig["tokenizer_class"].string_value();
        if (tokenizerClass == "PreTrainedTokenizerFast" || tokenizerClass == "LlamaTokenizerFast"
            || tokenizerClass == "Qwen2Tokenizer"
            || tokenizerClass == "BloomTokenizer") {
            // PreTrainedTokenizerFast
            std::string tokenizerFile = path + "tokenizer.json";
            auto tokenizer = json11::Json::parse(ReadAllFile(tokenizerFile), error);
            for (auto &it : tokenizer["model"]["vocab"].object_items()) {
                model->weight.AddTokenizerWord(it.first, it.second.int_value(), 1.0f);
            }
            std::map<std::string, int> spTokens;
            for (auto &it : tokenizer["added_tokens"].array_items()) {
                spTokens[it["content"].string_value()] = it["id"].int_value();
            }
            model->weight.tokenizer.SetSpecialTokens(spTokens);
            if (!spTokens.empty())
                model->weight.AddDict("tokenizer_has_special_tokens", "1");

            if (!tokenizer["decoder"].is_null() && !tokenizer["decoder"]["type"].is_null() && 
                tokenizer["decoder"]["type"].string_value() == "ByteLevel") {
                model->weight.tokenizer.byteAsChar = true;
                model->weight.AddDict("tokenizer_byte_as_char", "True");
            }
        } else if (tokenizerClass == "ChatGLM4Tokenizer") {
            // GLM4御用的分词
            std::vector <std::string> lines, line;
            SplitString(ReadAllFile(path + "tokenizer.model"), {'\r', '\n'}, lines);
            for (int i = 0; i < lines.size(); i++) {
                SplitString(lines[i], {' '}, line);
                model->weight.AddTokenizerWord(Base64Decode(line[0]), atoi(line[1].c_str()), 1.0f);
            }
            std::map<std::string, int> spTokens;
            for (auto &it : tokenizerConfig["added_tokens_decoder"].object_items()) {
                spTokens[it.second["content"].string_value()] = atoi(it.first.c_str());
            }
            model->weight.tokenizer.SetSpecialTokens(spTokens);
            model->weight.AddDict("tokenizer_has_special_tokens", "1");
            model->weight.AddDict("tokenizer_class", tokenizerClass);
            ((ChatGLMModel*)model)->tokenizerClass = tokenizerClass;

            // ChatGLM采用拼接token的方法，需要强行指定分割词的TokenID
            model->pre_prompt = "";
            model->user_role = ("<FLM_FIX_TOKEN_" + std::to_string(model->weight.tokenizer.GetTokenId("<|user|>"))  + ">\n");
            model->bot_role = ("<FLM_FIX_TOKEN_" + std::to_string(model->weight.tokenizer.GetTokenId("<|assistant|>")) + ">\n");
            model->history_sep = "";
            model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
            model->weight.tokenizer.chatTemplate = "";
        } else {
            ErrorInFastLLM("Unsupport tokenizer_class: " + tokenizerClass);
        }
    }

    // 从hf文件夹读取分词
    std::unique_ptr<basellm> CreateLLMTokenizerFromHF(const std::string &modelPath) {
        std::string error;
        std::string path = modelPath;
        if (path.back() != '/' || path.back() != '\\') {
            path += "/";
        }
        std::string configFile = path + "config.json";
        auto config = json11::Json::parse(ReadAllFile(configFile), error);
        basellm *model = CreateModelWithType(config["model_type"].string_value());
        LoadLLMTokenizerFromHFToModel(path, model);
        return std::unique_ptr<fastllm::basellm> (model);
    }

    // 将config中的内容递归地加入model->dict中
    void AddDictRecursion(basellm *model, const std::string &pre, const json11::Json &config) {
        for (auto &it : config.object_items()) {
            if (it.second.is_object()) {
                AddDictRecursion(model, pre + it.first + ".", it.second);
            } else {
                model->weight.AddDict(pre + it.first, it.second.is_string() ? it.second.string_value() : it.second.dump());
            }
        }
    }

    // 从hf文件夹读取，仅支持safetensor格式的模型
    std::unique_ptr <basellm> CreateLLMModelFromHF(const std::string &modelPath, 
                                                    DataType linearDataType, int groupCnt, bool skipTokenizer, const std::string &modelConfig) {
        bool isJsonModel = (modelConfig.size() > 0);

        std::string path = modelPath;
        if (path.back() != '/' || path.back() != '\\') {
            path += "/";
        }

        // 1. 检查是否有 model.safetensors.index.json,如果有就读取
        std::set <std::string> stFiles;
        std::string stIndexFile = path + "model.safetensors.index.json";
        std::string error;
        if (!FileExists(stIndexFile)) {
            stFiles.insert(path + "model.safetensors");
        } else {
            auto stIndex = json11::Json::parse(ReadAllFile(stIndexFile), error)["weight_map"];
            for (auto it : stIndex.object_items()) {
                stFiles.insert(path + it.second.string_value());
            }
        }
        SafeTensors safeTensors(stFiles);

        // 2. 创建网络基本信息
        std::string configFile = path + "config.json";
        auto config = json11::Json::parse(ReadAllFile(configFile), error);
        basellm *model = CreateModelWithType(isJsonModel ? "fastllmJson" : config["model_type"].string_value());
        if (isJsonModel) {
            ((GraphLLMModel*)model)->graphLLMModelConfig->Init(modelConfig);
        }
        AddDictRecursion(model, "", config);
        // 设置eos_token_id
        if (config["eos_token_id"].is_array()) {
            for (auto &it : config["eos_token_id"].array_items()) {
                model->eos_token_ids.insert(it.int_value());
            }
        } else {
            model->eos_token_id = config["eos_token_id"].int_value();
        }

        std::string generatetionConfigFile = path + "generation_config.json";
        if (FileExists(generatetionConfigFile)) {
            auto generation_config = json11::Json::parse(ReadAllFile(generatetionConfigFile), error);
            for (auto &it : generation_config.object_items()) {
                if ("eos_token_id" == it.first && it.second.type() == json11::Json::ARRAY)
                    continue;
                model->weight.AddDict(it.first, it.second.dump().c_str());
            }
            // 更新eos_token_id
            if (generation_config["eos_token_id"].is_array()) {
                for (auto &it : generation_config["eos_token_id"].array_items()) {
                    model->eos_token_ids.insert(it.int_value());
                }
            }
        }

        // 3. 读取分词
        if (!skipTokenizer) {
            LoadLLMTokenizerFromHFToModel(path, model);
        } else {
            DealLLMTokenizerFromHFToModel(path, model);
        }

        // 4.0 更新模型信息
        model->InitParams();

        // 4.1 读取权重
        auto tensors = safeTensors.GetSortedItemNames();
        
        // tensorMap[name]代表本名为name的tensor，创建后的名字以及类型
        // 有些tensor被共享，可能需要创建多次
        auto tensorMap = model->GetTensorMap(tensors);

        int cur = 0;
        long long totalBytes = 0;
        for (auto &tensorName : tensors) {
            auto &tensor = safeTensors.itmeDict[tensorName];
            auto oriDataType = DataType::FLOAT32;
            for (auto &it : tensorMap[tensorName]) {
                std::string weightName = it.first;
                auto dataType = it.second;
                if (dataType >= DATA_AUTO_NONE) {
                    // AUTO类型
                    dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;
                }
                if (it.second == DATA_AUTO_CONV) {
                    std::vector <int> realShape = tensor.intShape;
                    std::swap(realShape[0], realShape[1]);
                    model->weight.AddEmptyWeight(weightName, realShape, dataType);
                } else {
                    model->weight.AddEmptyWeight(weightName, tensor.intShape, dataType);
                }
            }

            totalBytes += tensor.bytes;
            printf("Load %d \r", (++cur) * 100 / (int)safeTensors.itmeDict.size());
            fflush(stdout);
        }

        // 4.2 读取
        std::vector <std::thread*> threads;
        int threadNum = std::min(16, std::max(4, (int)GetAlivePool()->threads.size()));
        int per = tensors.size() / threadNum;
        std::mutex locker;
        int cnt = 0;

        std::vector <std::pair <int, int> > parts;
        int start = 0;
        for (int i = 0; i < threadNum; i++) {
            int cur = start;
            long long now = 0;
            while (true) {
                if (now * threadNum >= totalBytes || start >= tensors.size()) {
                    break;
                }
                now += safeTensors.itmeDict[tensors[start]].bytes;
                start++;
            }
            parts.push_back(std::make_pair(cur, start));
        }
        parts.back().second = tensors.size();
        while (parts.size() < threadNum) {
            parts.push_back(std::make_pair(-1, -1));
        }

        for (int i = 0; i < threadNum; i++) {
            int st = per * i, end = (i == threadNum - 1) ? tensors.size() : per * (i + 1);
            threads.push_back(
                new std::thread([&](int st, int end) {
                    for (int i = st; i < end; i++) {
                        auto &tensorName = tensors[i];
                        auto &tensor = safeTensors.itmeDict[tensorName];

                        for (auto &it : tensorMap[tensorName]) {
                            auto oriDataType = DataType::FLOAT32;
                            std::string weightName = it.first;
                            auto dataType = it.second;
                            if (dataType >= DATA_AUTO_NONE) {
                                // AUTO类型
                                dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;
                            }
                            if (tensor.dtype == "BF16" &&
                                (dataType == DataType::FLOAT16 || dataType == DataType::INT8 || dataType == DataType::INT4_GROUP || dataType == DataType::INT4_NOZERO)) {
                                oriDataType = DataType::BFLOAT16;
                            }
                            if (tensor.dtype == "F16" && 
                                dataType == DataType::FLOAT16) {
                                oriDataType = DataType::FLOAT16;
                            }
                            tensor.CreateBuffer(oriDataType);
                            if (it.second == DATA_AUTO_CONV) {
                                tensor.Transpose(oriDataType);
                            }
                            model->weight[weightName].CreateFromOriData(WeightType::AUTO, oriDataType, tensor.buffer, groupCnt);
                            tensor.ClearBuffer();
                        }

                        locker.lock();
                        printf("Convert %d \r", (++cnt) * 100 / (int)tensorMap.size());
                        fflush(stdout);
                        locker.unlock();
                    }
                }, parts[i].first, parts[i].second)
            );
        }
        for (int i = 0; i < threads.size(); i++) {
            threads[i]->join();
            delete threads[i];
        }

        printf("\n");
        fflush(stdout);

        model->WarmUp();
        return std::unique_ptr<fastllm::basellm> (model);
    }
}
