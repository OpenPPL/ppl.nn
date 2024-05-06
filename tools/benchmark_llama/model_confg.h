#pragma once

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include "ppl/common/log.h"

#include <string>
#include <fstream>

struct ModelConfig final {
    std::string model_type;
    std::string model_dir;
    std::string model_param_path;
    std::string quant_method;

    int32_t tensor_parallel_size = 0;

    int32_t hidden_dim;
    int32_t intermediate_dim;
    int32_t num_layers;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t vocab_size;

    float norm_eps; // not used

    int32_t cache_quant_bit;
    int32_t cache_quant_group;
    int32_t cache_layout;
    int32_t cache_mode;
    int32_t page_size;

    bool dynamic_batching;
    bool auto_causal;

    bool ParseModelParam() {
        std::ifstream ifs(model_param_path);
        rapidjson::IStreamWrapper isw(ifs);
        rapidjson::Document document;
        if (document.ParseStream(isw) == false) {
            LOG(ERROR) << "ParseStream failed";
            return false;
        }
        document.ParseStream(isw);

        auto it = document.FindMember("num_heads");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [num_heads] failed";
            return false;
        }
        num_heads = it->value.GetInt();

        it = document.FindMember("num_kv_heads");
        if (it == document.MemberEnd()) {
            num_kv_heads = num_heads;
        } else {
            num_kv_heads = it->value.GetInt();
        }

        it = document.FindMember("num_layers");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [num_layers] failed";
            return false;
        }
        num_layers = it->value.GetInt();

        it = document.FindMember("hidden_dim");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [hidden_dim] failed";
            return false;
        }
        hidden_dim = it->value.GetInt();

        it = document.FindMember("intermediate_dim");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [intermediate_dim] failed";
            return false;
        }
        intermediate_dim = it->value.GetInt();

        it = document.FindMember("vocab_size");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [vocab_size] failed";
            return false;
        }
        vocab_size = it->value.GetInt();

        it = document.FindMember("cache_quant_bit");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [cache_quant_bit] failed";
            return false;
        }
        cache_quant_bit = it->value.GetInt();

        it = document.FindMember("cache_quant_group");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [cache_quant_group] failed";
            return false;
        }
        cache_quant_group = it->value.GetInt();

        it = document.FindMember("cache_layout");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [cache_layout] failed";
            return false;
        }
        cache_layout = it->value.GetInt();

        it = document.FindMember("cache_mode");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [cache_mode] failed";
            return false;
        }
        cache_mode = it->value.GetInt();

        it = document.FindMember("dynamic_batching");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [dynamic_batching] failed";
            return false;
        }
        dynamic_batching = it->value.GetBool();

        it = document.FindMember("auto_causal");
        if (it == document.MemberEnd()) {
            LOG(ERROR) << "find key [auto_causal] failed";
            return false;
        }
        auto_causal = it->value.GetBool();

        page_size = 0;
        if (cache_mode == 1) {
            it = document.FindMember("page_size");
            if (it == document.MemberEnd()) {
                LOG(ERROR) << "find key [page_size] failed";
                return false;
            }
            page_size = it->value.GetInt();
        }

        LOG(INFO) << "num_layers = " << num_layers;
        LOG(INFO) << "num_heads = " << num_heads;
        LOG(INFO) << "num_kv_heads = " << num_kv_heads;
        LOG(INFO) << "hidden_dim = " << hidden_dim;
        LOG(INFO) << "intermediate_dim = " << intermediate_dim;
        LOG(INFO) << "vocab_size = " << vocab_size;

        LOG(INFO) << "cache_quant_bit = " << cache_quant_bit;
        LOG(INFO) << "cache_quant_group = " << cache_quant_group;
        LOG(INFO) << "cache_layout = " << cache_layout;
        LOG(INFO) << "cache_mode = " << cache_mode;
        LOG(INFO) << "page_size = " << page_size;

        LOG(INFO) << "dynamic_batching = " << dynamic_batching;
        LOG(INFO) << "auto_causal = " << auto_causal;

        return true;
    }
};
