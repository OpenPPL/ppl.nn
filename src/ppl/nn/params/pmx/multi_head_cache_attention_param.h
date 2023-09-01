#ifndef _ST_HPC_PPL_NN_PARAMS_PMX_MULTI_HEAD_CACHE_ATTENTION_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_PMX_MULTI_HEAD_CACHE_ATTENTION_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace pmx {

struct MultiHeadCacheAttentionParam final : public ir::TypedAttr<MultiHeadCacheAttentionParam> {
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    bool is_causal;
    int32_t num_layer;
    int32_t layer_idx;
    int32_t quant_bit;
    int32_t quant_group;
    int32_t cache_mode;
    int32_t cache_layout;

    bool operator==(const MultiHeadCacheAttentionParam& p) const {
        return (num_heads == p.num_heads
            && num_kv_heads == p.num_kv_heads
            && head_dim == p.head_dim
            && is_causal == p.is_causal
            && num_layer == p.num_layer
            && layer_idx == p.layer_idx
            && quant_bit == p.quant_bit
            && quant_group == p.quant_group
            && cache_mode == p.cache_mode
            && cache_layout == p.cache_layout);
    }
};

}}} // namespace ppl::nn::pmx

#endif
