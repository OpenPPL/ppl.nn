#pragma once

#include "ppl/common/retcode.h"

#include <cuda_runtime.h>

class CudaSampler final {
public:
    ppl::common::RetCode Init(cudaStream_t stream) {
        stream_ = stream;
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode Clear();

    ppl::common::RetCode SampleArgMax(
        const float* logits_device,
        const int32_t batch,
        const int32_t vocab_size,
        const int32_t batch_stride,
        int32_t* output_host);

private:
    cudaStream_t stream_ = 0;
    int32_t max_batch_ = 0;
    int32_t* output_device_ = nullptr;
};
