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
        std::string cublas_layout_hint;
    };

    CudaTextGenerator(const ConstructOptions& options) {
        cublas_layout_hint_ = options.cublas_layout_hint;
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
    std::string cublas_layout_hint_;
    std::unique_ptr<CudaSampler> sampler_;

    void InitCudaThread();
    void FinalizeCudaThread();

#ifdef PPLNN_CUDA_ENABLE_NCCL
    void InitNccl();
    void FinalizeNccl();
#endif
};
