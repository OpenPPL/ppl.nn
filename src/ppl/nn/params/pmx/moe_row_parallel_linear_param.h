#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_MOE_ROW_PARALLEL_LINEAR_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_MOE_ROW_PARALLEL_LINEAR_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct MoeRowParallelLinearParam final : public ir::TypedAttr<MoeRowParallelLinearParam> {
    int32_t num_experts;
    int32_t in_features;
    int32_t out_features;
    bool bias_term;
    bool input_is_parallel;

    bool operator==(const MoeRowParallelLinearParam& p) const {
        return (num_experts == p.num_experts 
            && in_features == p.in_features 
            && out_features == p.out_features 
            && bias_term == p.bias_term 
            && input_is_parallel == p.input_is_parallel);
    }
};

}}} // namespace ppl::nn::pmx

#endif
