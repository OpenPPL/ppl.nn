#ifndef PPLCUDA_KERNEL_INCLUDE_GATHERND_GATHERND_H_
#define PPLCUDA_KERNEL_INCLUDE_GATHERND_GATHERND_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

int64_t pplGatherNDGetTempBufferSize(
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices);

ppl::common::RetCode PPLCUDAGatherNDForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* indices_shape,
    const void* indices,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int batch_dims = 0);

#endif //PPLCUDA_KERNEL_INCLUDE_GATHERND_GATHERND_H_
