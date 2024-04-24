#include "ppl/nn/models/onnx/parsers/opmx/parse_multi_head_cache_attention_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace opmx {

RetCode ParseMultiHeadCacheAttentionParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node, ir::Attr* arg) {
    auto param = static_cast<ppl::nn::opmx::MultiHeadCacheAttentionParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "num_heads", &param->num_heads, -1)) {
        LOG(ERROR) << node->GetName() << ": missing num_heads";
        return RC_INVALID_VALUE;
    }

    if (!onnx::utils::GetNodeAttr(pb_node, "head_dim", &param->head_dim, -1)) {
        LOG(ERROR) << node->GetName() << ": missing head_dim";
        return RC_INVALID_VALUE;
    }

    if (!onnx::utils::GetNodeAttr(pb_node, "num_layer", &param->num_layer, -1)) {
        LOG(ERROR) << node->GetName() << ": missing num_layer";
        return RC_INVALID_VALUE;
    }

    if (!onnx::utils::GetNodeAttr(pb_node, "layer_idx", &param->layer_idx, -1)) {
        LOG(ERROR) << node->GetName() << ": missing layer_idx";
        return RC_INVALID_VALUE;
    }

    int32_t is_causal;
    onnx::utils::GetNodeAttr(pb_node, "is_causal", &is_causal, 0);
    param->is_causal = is_causal;
    onnx::utils::GetNodeAttr(pb_node, "num_kv_heads", &param->num_kv_heads, param->num_heads);
    onnx::utils::GetNodeAttr(pb_node, "quant_bit", &param->quant_bit, 0);
    onnx::utils::GetNodeAttr(pb_node, "quant_group", &param->quant_group, 8);
    onnx::utils::GetNodeAttr(pb_node, "cache_mode", &param->cache_mode, 0);
    onnx::utils::GetNodeAttr(pb_node, "cache_layout", &param->cache_layout, 0);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx