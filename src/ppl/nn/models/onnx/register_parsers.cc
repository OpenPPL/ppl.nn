#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/param_parser_manager.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_column_parallel_linear_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_key_value_cache_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_multi_head_attention_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_multi_head_cache_attention_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_parallel_embedding_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_rotary_position_embedding_param.h"
#include "ppl/nn/models/onnx/parsers/pmx/parse_row_parallel_linear_param.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

template <typename T>
shared_ptr<ir::Attr> CreateParam() {
    return make_shared<T>();
}

#define PPL_REGISTER_OP_WITH_PARAM(domain, type, first_version, last_version, param_type, parse_param_func,    \
                                   pack_param_func)                                                            \
    do {                                                                                                       \
        if (last_version < first_version) {                                                                    \
            LOG(ERROR) << "register op[" << domain << ":" << type << "] failed: last_version[" << last_version \
                       << "] < first_version[" << first_version << "]";                                        \
            exit(-1);                                                                                          \
        }                                                                                                      \
                                                                                                               \
        onnx::ParserInfo parse_info;                                                                           \
        parse_info.create_param = CreateParam<param_type>;                                                     \
        parse_info.parse_param = parse_param_func;                                                             \
        parse_info.pack_param = pack_param_func;                                                               \
        auto status = onnx::ParamParserManager::GetInstance()->Register(                                       \
            domain, type, utils::VersionRange(first_version, last_version), parse_info);                       \
        if (status != RC_SUCCESS) {                                                                            \
            exit(-1);                                                                                          \
        }                                                                                                      \
    } while (0)

#define PPL_REGISTER_OP_WITHOUT_PARAM(domain, type, first_version, last_version, parse_param_func) \
    do {                                                                                           \
        onnx::ParserInfo parse_info;                                                               \
        parse_info.create_param = nullptr;                                                         \
        parse_info.parse_param = parse_param_func;                                                 \
        parse_info.pack_param = nullptr;                                                           \
        onnx::ParamParserManager::GetInstance()->Register(                                         \
            domain, type, utils::VersionRange(first_version, last_version), parse_info);           \
    } while (0)

// NOTE: sorted in alphabet order
void RegisterParsers() {
    static bool st_registered = false;
    if (st_registered) {
        return;
    }
    st_registered = true;

    PPL_REGISTER_OP_WITH_PARAM("pmx", "ColumnParallelLinear", 1, 1, ppl::nn::pmx::ColumnParallelLinearParam,
                               ppl::nn::pmx::ParseColumnParallelLinearParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx", "KeyValueCache", 1, 1, ppl::nn::pmx::KeyValueCacheParam,
                               ppl::nn::pmx::ParseKeyValueCacheParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx", "MultiHeadAttention", 1, 1, ppl::nn::pmx::MultiHeadAttentionParam,
                               ppl::nn::pmx::ParseMultiHeadAttentionParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx", "ParallelEmbedding", 1, 1, ppl::nn::pmx::ParallelEmbeddingParam,
                               ppl::nn::pmx::ParseParallelEmbeddingParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx", "RotaryPositionEmbedding", 1, 1, ppl::nn::pmx::RotaryPositionEmbeddingParam,
                               ppl::nn::pmx::ParseRotaryPositionEmbeddingParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx", "Rotary2DPositionEmbedding", 1, 1, ppl::nn::pmx::RotaryPositionEmbeddingParam,
                               ppl::nn::pmx::ParseRotaryPositionEmbeddingParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx", "RowParallelLinear", 1, 1, ppl::nn::pmx::RowParallelLinearParam,
                               ppl::nn::pmx::ParseRowParallelLinearParam, nullptr);

    // dynamic batching
    PPL_REGISTER_OP_WITH_PARAM("pmx.dynamic_batching", "KeyValueCache", 1, 1, ppl::nn::pmx::KeyValueCacheParam,
                               ppl::nn::pmx::ParseKeyValueCacheParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx.dynamic_batching", "MultiHeadAttention", 1, 1,
                               ppl::nn::pmx::MultiHeadAttentionParam, ppl::nn::pmx::ParseMultiHeadAttentionParam,
                               nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx.dynamic_batching", "MultiHeadCacheAttention", 1, 1,
                               ppl::nn::pmx::MultiHeadCacheAttentionParam, ppl::nn::pmx::ParseMultiHeadCacheAttentionParam,
                               nullptr);
    PPL_REGISTER_OP_WITH_PARAM("pmx.dynamic_batching", "RotaryPositionEmbedding", 1, 1, ppl::nn::pmx::RotaryPositionEmbeddingParam,
                               ppl::nn::pmx::ParseRotaryPositionEmbeddingParam, nullptr);

    PPL_REGISTER_OP_WITH_PARAM("pmx.dynamic_batching", "Rotary2DPositionEmbedding", 1, 1, ppl::nn::pmx::RotaryPositionEmbeddingParam,
                               ppl::nn::pmx::ParseRotaryPositionEmbeddingParam, nullptr);
}

}}} // namespace ppl::nn::onnx
