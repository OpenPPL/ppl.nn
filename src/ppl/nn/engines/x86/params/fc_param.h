#ifndef _ST_HPC_PPL_NN_ENGINES_X86_PARAMS_FC_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_PARAMS_FC_PARAM_H_

#include "ppl/kernel/x86/fp32/fc.h"

namespace ppl { namespace nn { namespace x86 {

struct FCParam {
    ppl::kernel::x86::fc_fp32_param param;
    ppl::kernel::x86::fc_fp32_algo_info algo_info;
    ppl::kernel::x86::fc_fp32_manager* mgr = nullptr;
};

}}}; // namespace ppl::nn::x86

#endif