#ifndef _ST_HPC_PPL_NN_PARAMS_OPMX_ROTARY_EMBED_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_OPMX_ROTARY_EMBED_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace opmx {

struct RotaryPositionEmbeddingParam final : public ir::TypedAttr<RotaryPositionEmbeddingParam> {
    int32_t bypass_key;
    int32_t rotary_dim;
    float theta;

    bool operator==(const RotaryPositionEmbeddingParam& p) const {
        return (bypass_key == p.bypass_key
            && fabs(theta - p.theta) <= 1e-05
            && rotary_dim == p.rotary_dim);
    }
};

}}} // namespace ppl::nn::opmx

#endif
