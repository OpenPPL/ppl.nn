#ifndef PPLCUDA_KERNEL_INCLUDE_PAD_PAD_H_
#define PPLCUDA_KERNEL_INCLUDE_PAD_PAD_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/params/onnx/pad_param.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

#define PAD_PARAM_MAX_DIM_SIZE 5
struct PadKernelParam {
    typedef uint32_t pad_mode_t;
    enum { PAD_MODE_CONSTANT = 0, PAD_MODE_REFLECT = 1, PAD_MODE_EDGE = 2 };

    float constant_value = 0.f;
    pad_mode_t mode = PAD_MODE_CONSTANT;
};

ppl::common::RetCode PPLCUDAPadForwardImp(
    cudaStream_t stream,
    PadKernelParam param,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* pads_shape,
    const int64_t* pads,
    ppl::nn::TensorShape* output_shape,
    void* output);

#endif //PPLCUDA_KERNEL_INCLUDE_PAD_PAD_H_