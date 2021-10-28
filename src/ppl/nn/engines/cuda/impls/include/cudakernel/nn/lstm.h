#include <cuda.h>
#include "cudakernel/common/rnn_common.h"

#include "ppl/nn/engines/cuda/module/cuda_module.h"

int64_t PPLCUDALstmGetRuntimeBufSize(
    const ppl::nn::TensorShape *X_shape,
    const unsigned int direction,
    const int64_t hidden_size);

ppl::common::RetCode PPLCUDALstmForwardImp(
    cudaStream_t stream,
    ppl::nn::cuda::CUDAModule *module,
    const ppl::nn::TensorShape *X_shape,
    const void *X,
    const void *X_weight,
    const void *R_weight,
    const void *P_weight,
    const void *bias,
    const void *sequence_lens,
    const void *initial_h,
    const void *initial_c,
    const unsigned int direction,
    const int64_t hidden_size,
    void *temp_buffer,
    void *Y,
    void *Y_h,
    void *Y_c);
