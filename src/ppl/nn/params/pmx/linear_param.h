#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_LINEAR_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_LINEAR_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct LinearParam final : public ir::TypedAttr<LinearParam> {
    int32_t in_features;
    int32_t out_features;
    bool bias_term;

    bool operator==(const LinearParam& p) const {
        return (in_features == p.in_features
            && out_features == p.out_features
            && bias_term == p.bias_term);
    }
};

}}} // namespace ppl::nn::pmx

#endif