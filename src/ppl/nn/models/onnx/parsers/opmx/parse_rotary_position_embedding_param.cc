#include "ppl/nn/models/onnx/parsers/opmx/parse_rotary_position_embedding_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace opmx {

RetCode ParseRotaryPositionEmbeddingParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node, ir::Attr* arg) {
    auto param = static_cast<ppl::nn::opmx::RotaryPositionEmbeddingParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "theta", &param->theta, -1)) {
        LOG(ERROR) << node->GetName() << ": missing theta";
        return RC_INVALID_VALUE;
    }

    onnx::utils::GetNodeAttr(pb_node, "bypass_key", &param->bypass_key, 0);
    onnx::utils::GetNodeAttr(pb_node, "rotary_dim", &param->rotary_dim, 0);
    onnx::utils::GetNodeAttr(pb_node, "scaling_factor", &param->scaling_factor, 1.0f);
    onnx::utils::GetNodeAttr(pb_node, "max_position_embeddings", &param->max_position_embeddings, 2048);

    string scaling_type;
    onnx::utils::GetNodeAttr(pb_node, "scaling_type", &scaling_type, "");

    if (scaling_type == "") {
        param->scaling_type = ppl::nn::opmx::RotaryPositionEmbeddingParam::SCALING_TYPE_NONE;
    } else if (scaling_type == "linear") {
        param->scaling_type = ppl::nn::opmx::RotaryPositionEmbeddingParam::SCALING_TYPE_LINEAR;
    } else if (scaling_type == "dynamic") {
        param->scaling_type = ppl::nn::opmx::RotaryPositionEmbeddingParam::SCALING_TYPE_DYNAMIC;
    } else {
        LOG(ERROR) << "unsupported scaling_type: " << scaling_type;
        return RC_UNSUPPORTED;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx