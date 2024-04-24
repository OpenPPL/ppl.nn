#ifndef _ST_HPC_PPL_NN_PARAMS_OPMX_MULTI_HEAD_ATTENTION_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_OPMX_MULTI_HEAD_ATTENTION_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace opmx {

struct MultiHeadAttentionParam final : public ir::TypedAttr<MultiHeadAttentionParam> {
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    bool is_causal;

    bool operator==(const MultiHeadAttentionParam& p) const {
        return (num_heads == p.num_heads
            && num_kv_heads == p.num_kv_heads
            && head_dim == p.head_dim
            && is_causal == p.is_causal);
    }
};

}}} // namespace ppl::nn::opmx

#endif
