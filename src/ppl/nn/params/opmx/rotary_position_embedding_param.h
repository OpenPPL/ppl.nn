#ifndef _ST_HPC_PPL_NN_PARAMS_OPMX_ROTARY_EMBED_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_OPMX_ROTARY_EMBED_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace opmx {

struct RotaryPositionEmbeddingParam final : public ir::TypedAttr<RotaryPositionEmbeddingParam> {
    enum {
        SCALING_TYPE_NONE = 0,
        SCALING_TYPE_LINEAR = 1,
        SCALING_TYPE_DYNAMIC = 2,
    };

    int32_t bypass_key;
    int32_t rotary_dim;
    float theta;
    int32_t max_position_embeddings;
    int32_t scaling_type;
    float scaling_factor;

    bool operator==(const RotaryPositionEmbeddingParam& p) const {
        return (bypass_key == p.bypass_key
            && fabs(theta - p.theta) <= 1e-05
            && rotary_dim == p.rotary_dim
            && scaling_type == p.scaling_type
            && (scaling_type == SCALING_TYPE_NONE || max_position_embeddings == p.max_position_embeddings)
            && (scaling_type == SCALING_TYPE_NONE || fabs(scaling_factor - p.scaling_factor) <= 1e-05));
    }
};

}}} // namespace ppl::nn::opmx

#endif
