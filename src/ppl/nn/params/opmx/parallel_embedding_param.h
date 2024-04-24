#ifndef _ST_HPC_PPL_NN_PARAMS_OPMX_PARALLEL_EMBEDDING_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_OPMX_PARALLEL_EMBEDDING_PARAM_H_

#include "ppl/nn/ir/attr.h"
#include <stdint.h>
#include <cmath>

namespace ppl { namespace nn { namespace opmx {

struct ParallelEmbeddingParam final : public ir::TypedAttr<ParallelEmbeddingParam> {
    int32_t num_embeddings;
    int32_t embedding_dims;
    int32_t padding_idx;
    float max_norm;
    float norm_type;

    bool operator==(const ParallelEmbeddingParam& p) const {
        return (num_embeddings == p.num_embeddings
            && embedding_dims == p.embedding_dims
            && padding_idx == p.padding_idx
            && max_norm == p.max_norm
            && norm_type == p.norm_type);
    }
};

}}} // namespace ppl::nn::opmx

#endif
