#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"


template<typename TPar>
ppl::common::RetCode PPLCUDALayernormForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* output_shape,
    void* output,
    const TPar* alpha, 
    const TPar* beta,
    const int eps);