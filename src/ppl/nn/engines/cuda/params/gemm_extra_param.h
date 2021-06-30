#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_PARAMS_GEMM_EXTRA_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_PARAMS_GEMM_EXTRA_PARAM_H_

#include "ppl/nn/engines/cuda/params/conv_extra_param.h"
#include "ppl/nn/params/onnx/gemm_param.h"

namespace ppl { namespace nn { namespace cuda {

struct GemmExtraParam {
    uint32_t kernel_index = 0;
    uint32_t has_activation = 0;
    bool has_clip = false;
    ClipParam clip;
};

struct CudaGemmParam {
    ppl::nn::common::GemmParam param;
    GemmExtraParam extra_param;
};

}}} // namespace ppl::nn::cuda

#endif
