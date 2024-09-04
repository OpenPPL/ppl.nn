#ifdef PPLNN_ENABLE_FP8

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_PASSES_F8F8_CAST_PASS_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_PASSES_F8F8_CAST_PASS_H_

#include "ppl/nn/engines/llm_cuda/opt_pass.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace f8f8 {

OptPassStatus CastPass(const OptKernelOptions& options);

}}}}}

#endif
#endif

