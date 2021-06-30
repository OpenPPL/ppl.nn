#ifndef PPLCUDA_KERNEL_INCLUDE_SLICE_SLICE_H_
#define PPLCUDA_KERNEL_INCLUDE_SLICE_SLICE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <stdint.h>

#define SLICE_PARAM_MAX_DIM_SIZE  5
struct SliceKernelParam {
    int64_t axes_num = 0;
    int64_t starts[SLICE_PARAM_MAX_DIM_SIZE];
    int64_t ends[SLICE_PARAM_MAX_DIM_SIZE];
    int64_t axes[SLICE_PARAM_MAX_DIM_SIZE];
    int64_t steps[SLICE_PARAM_MAX_DIM_SIZE];
};

ppl::common::RetCode PPLCUDASliceForwardImp(
    cudaStream_t stream,
    SliceKernelParam param,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_SLICE_SLICE_H_