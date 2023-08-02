#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_LAYER_NORM_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_LAYER_NORM_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct LayerNormParam final : public ir::TypedAttr<LayerNormParam> {
    bool elementwise_affine;
    int32_t axis;
    float eps;
    bool skip_term;

    bool operator==(const LayerNormParam& p) const {
        return (elementwise_affine == p.elementwise_affine && axis == p.axis && fabs(eps - p.eps) <= 1e-05 &&
                skip_term == p.skip_term);
    }
};

}}} // namespace ppl::nn::pmx

#endif
