#include "ppl/nn/models/onnx/parsers/pmx/parse_multi_head_attention_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ParseMultiHeadAttentionParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node, ir::Attr* arg) {
    auto param = static_cast<ppl::nn::pmx::MultiHeadAttentionParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "num_heads", &param->num_heads, -1)) {
        LOG(ERROR) << node->GetName() << ": missing num_heads";
        return RC_INVALID_VALUE;
    }

    if (!onnx::utils::GetNodeAttr(pb_node, "head_dim", &param->head_dim, -1)) {
        LOG(ERROR) << node->GetName() << ": missing head_dim";
        return RC_INVALID_VALUE;
    }

    int32_t is_causal;
    onnx::utils::GetNodeAttr(pb_node, "num_kv_heads", &param->num_kv_heads, param->num_heads);
    onnx::utils::GetNodeAttr(pb_node, "is_causal", &is_causal, 0);
    param->is_causal = is_causal;
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx