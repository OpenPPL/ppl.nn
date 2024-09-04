#pragma once

#include "../text_generator.h"
#include "cuda_sampler.h"

#ifdef PPLNN_CUDA_ENABLE_NCCL
#include <nccl.h>
#else
typedef void* ncclComm_t;
#endif

class CudaTextGenerator final : public TextGenerator {
public:
    struct ConstructOptions {
        std::string cublas_layout_hint = "default";
        bool disable_graph_fusion = false;
        bool enable_cache_prefill = false;
        bool disable_decoding_shm_mha = false;
        bool disable_decoding_inf_mha = false;
        bool disable_decoding_inf_gqa = false;
        uint32_t configure_decoding_attn_split_k = 1;
        uint32_t specify_decoding_attn_tpb = 0;
    };

    CudaTextGenerator(const ConstructOptions& options) {
        construct_options_ = options;
    }

    virtual bool CheckParameters() override;
    virtual ppl::common::RetCode DoInit() override;
    virtual ppl::common::RetCode DoFinalize() override;
    virtual uint64_t GetMemUsage() override;

    virtual ppl::common::RetCode CreateSampler() override;
    virtual ppl::common::RetCode DestorySampler() override;

    virtual ppl::nn::Engine* ThreadCreateEngine(const int32_t tid) override;
    virtual void ThreadExtractTensors(const int32_t tid) override;
    virtual ppl::common::RetCode ThreadReallocKVCache(const int32_t tid) override;
    virtual ppl::common::RetCode ThreadFreeKVCache(const int32_t tid) override;
    virtual ppl::common::RetCode ThreadSampleArgMax(
        const int32_t tid,
        const float* logits,
        const int32_t batch,
        const int32_t vocab_size,
        const int32_t batch_stride,
        int32_t* output) override;

private:
    std::vector<ncclComm_t> nccl_comm_list_;
    std::vector<std::unique_ptr<ppl::nn::DeviceContext>> host_device_list_;
    std::unique_ptr<CudaSampler> sampler_;

    ConstructOptions construct_options_;

    void InitCudaThread();
    void FinalizeCudaThread();

#ifdef PPLNN_CUDA_ENABLE_NCCL
    void InitNccl();
    void FinalizeNccl();
#endif
};
