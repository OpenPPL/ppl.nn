#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_MOE_REDUCE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_MOE_REDUCE_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct MoeReduceParam final : public ir::TypedAttr<MoeReduceParam> {
    int32_t num_experts_per_token;

    bool operator==(const MoeReduceParam& p) const {
        return (num_experts_per_token == p.num_experts_per_token);
    }
};

}}} // namespace ppl::nn::pmx

#endif