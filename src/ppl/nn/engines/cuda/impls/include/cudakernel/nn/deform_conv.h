#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/engines/cuda/module/cuda_module.h"

#include "ppl/common/retcode.h"

int64_t PPLCUDADeformConvGetBufSize(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *flt_shape,
    const ppl::nn::TensorShape *output_shape);

ppl::common::RetCode PPLCUDADeformConvForward(
    const cudaStream_t &stream,
    ppl::nn::cuda::CUDAModule *module,
    const ppl::nn::TensorShape *output_shape,
    const ppl::nn::TensorShape *input_shape,
    void *output,
    const void *input,
    const void *flt,
    const void *offset,
    const void *mask,
    const void *bias,
    const int group,
    const int offset_group,
    const int channels,
    const int num_output,
    const int stride_h,
    const int stride_w,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    bool use_mask,
    void *temp_buffer);

ppl::common::RetCode PPLCUDADeformConvModifyWeights(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape *flt_shape,
    const void *in_flt,
    void *out_flt);
