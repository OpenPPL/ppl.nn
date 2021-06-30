#ifndef _ST_HPC_PPL_NN_ENGINES_X86_X86_COMMON_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_X86_COMMON_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace x86 {

struct X86CommonParam {
    std::vector<ppl::common::dataformat_t> output_formats;
};

}}} // namespace ppl::nn::x86

#endif
