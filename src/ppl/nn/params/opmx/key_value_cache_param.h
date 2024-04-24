#ifndef _ST_HPC_PPL_NN_PARAMS_OPMX_KEY_VALUE_CACHE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_OPMX_KEY_VALUE_CACHE_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace opmx {

struct KeyValueCacheParam final : public ir::TypedAttr<KeyValueCacheParam> {
    int32_t num_layer;
    int32_t layer_idx;
    int32_t quant_bit;
    int32_t quant_group;
    int32_t num_repeat;
    int32_t cache_mode;
    int32_t cache_layout;

    bool operator==(const KeyValueCacheParam& p) const {
        return (num_layer == p.num_layer
            && layer_idx == p.layer_idx
            && quant_bit == p.quant_bit
            && quant_group == p.quant_group
            && num_repeat == p.num_repeat
            && cache_mode == p.cache_mode
            && cache_layout == p.cache_layout);
    }
};

}}} // namespace ppl::nn::opmx

#endif
