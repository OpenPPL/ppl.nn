#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_PARAMS_CONCAT_EXTRA_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_PARAMS_CONCAT_EXTRA_PARAM_H_

#include "ppl/nn/oputils/onnx/reshape_concat.h"

namespace ppl { namespace nn { namespace cuda {

struct ConcatExtraParam {
    uint32_t mask = 0;
};

struct CudaConcatParam {
    ppl::nn::common::ConcatParam param;
    ConcatExtraParam extra_param;
};

}}} // namespace ppl::nn::cuda

#endif
