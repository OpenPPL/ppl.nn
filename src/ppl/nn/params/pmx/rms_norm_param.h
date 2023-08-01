#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_RMS_NORM_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_RMS_NORM_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct RMSNormParam final : public ir::TypedAttr<RMSNormParam> {
    int32_t axis;
    float eps;
    bool skip_term;

    bool operator==(const RMSNormParam& p) const {
        return (axis == p.axis && fabs(eps - p.eps) <= 1e-05 && skip_term == p.skip_term);
    }
};

}}} // namespace ppl::nn::pmx

#endif
