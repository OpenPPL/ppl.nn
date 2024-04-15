#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_MOE_SELECT_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_MOE_SELECT_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct MoeSelectParam final : public ir::TypedAttr<MoeSelectParam> {
    int32_t num_experts;
    int32_t num_experts_per_token;

    bool operator==(const MoeSelectParam& p) const {
        return (num_experts == p.num_experts
            && num_experts_per_token == p.num_experts_per_token);
    }
};

}}} // namespace ppl::nn::pmx

#endif