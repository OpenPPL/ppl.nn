#pragma once

#include <vector>
#include <memory>

#include "ppl/nn/runtime/options.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/engines/engine.h"

#include "thread_pool.h"
#include "generation_state.h"
#include "profiler.h"

class TextGenerator {
public:
    ppl::common::RetCode InitModel(
        const std::string &model_type,
        const std::string &model_dir,
        const std::string &model_param_path,
        const std::string &quant_method,
        const int32_t tensor_parallel_size,
        const bool use_pmx_format
    );
    ppl::common::RetCode FinalizeModel();

    ppl::common::RetCode PrepareGeneration(
        const std::vector<Request>& requests,
        std::vector<Response> *response
    );
    ppl::common::RetCode Generate(
        const std::vector<Request>& requests,
        std::vector<Response> *response,
        Profiler *optinal_profiler = nullptr
    );

    const ModelConfig& GetModelConfig() const {
        return model_config_;
    }

protected:
    struct RuntimeThreadArg final {
        std::unique_ptr<ppl::nn::Runtime> runtime;

        void *kv_cache_ptr = nullptr;
        void *kv_scale_ptr = nullptr;

        ppl::nn::Tensor* token_ids;
        ppl::nn::Tensor* attn_mask;
        ppl::nn::Tensor* seq_starts;
        ppl::nn::Tensor* kv_starts;
        ppl::nn::Tensor* cache_indices;
        ppl::nn::Tensor* decoding_batches;
        ppl::nn::Tensor* start_pos;
        ppl::nn::Tensor* max_seq_len;
        ppl::nn::Tensor* max_kv_len;
        ppl::nn::Tensor* kv_cache;
        ppl::nn::Tensor* kv_scale;

        ppl::nn::Tensor* logits;
    };

    ModelConfig model_config_;
    std::vector<std::unique_ptr<ppl::nn::Engine>> engine_list_;
    std::vector<RuntimeThreadArg> runtime_thread_args_;
    GenerationState state_;

    virtual ppl::common::RetCode DoInit() = 0;
    virtual ppl::common::RetCode DoFinalize() = 0;
    virtual bool CheckParameters() = 0;
    virtual uint64_t GetMemUsage() = 0;

    virtual ppl::common::RetCode CreateSampler() = 0;
    virtual ppl::common::RetCode DestorySampler() = 0;

    /*
        All methods start with "Thread" means it must be called in RuntimePoolRun()'s working function
        RuntimePoolRun([](uint32_t n, uint32_t i) {
            ThreadDoSomething(i);
        });

        And other methods without "Thread" prefix must not be called in RuntimePoolRun()'s working function
    */
    virtual ppl::nn::Engine* ThreadCreateEngine(const int32_t tid) = 0;
    virtual void ThreadExtractTensors(const int32_t tid) = 0;
    virtual ppl::common::RetCode ThreadReallocKVCache(const int32_t tid) = 0;
    virtual ppl::common::RetCode ThreadFreeKVCache(const int32_t tid) = 0;
    virtual ppl::common::RetCode ThreadSampleArgMax(
        const int32_t tid,
        const float* logits, // [>=batch, batch_stride(>=vocab_size)]
        const int32_t batch,
        const int32_t vocab_size,
        const int32_t batch_stride,
        int32_t* output) = 0; // [>=batch]
   

private:
    ThreadPool runtime_thread_pool_;
    int64_t alloced_cache_length_ = 0;

    ppl::nn::Runtime* ThreadCreateRuntime(const int32_t tid, const bool use_pmx_format);
    bool ThreadSetInputTensors(const int32_t tid);

protected:
    void RuntimePoolRun(const std::function<ppl::common::RetCode(uint32_t nr_threads, uint32_t thread_idx)>& f) {
        runtime_thread_pool_.Run(f);
    }

};
