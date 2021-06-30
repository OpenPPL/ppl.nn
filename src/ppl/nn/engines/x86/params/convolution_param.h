#ifndef _ST_HPC_PPL_NN_ENGINES_X86_PARAMS_CONVOLUTION_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_PARAMS_CONVOLUTION_PARAM_H_

#include <functional>

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

namespace ppl { namespace nn { namespace x86 {

struct Convolution2DParam {
    ppl::kernel::x86::conv2d_fp32_param param;
    ppl::kernel::x86::conv2d_fp32_algo_info algo_info;
    ppl::kernel::x86::conv2d_fp32_manager* mgr = nullptr;
    ppl::kernel::x86::conv2d_fp32_manager* fallback_mgr = nullptr;
    std::function<bool(const TensorImpl*, const TensorImpl*, const ppl::kernel::x86::conv2d_fp32_param*)>
        infer_fallback_func;
};

}}}; // namespace ppl::nn::x86

#endif