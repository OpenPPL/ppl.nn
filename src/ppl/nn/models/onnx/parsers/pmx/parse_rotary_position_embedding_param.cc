#include "ppl/nn/models/onnx/parsers/pmx/parse_rotary_position_embedding_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ParseRotaryPositionEmbeddingParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node, ir::Attr* arg) {
    auto param = static_cast<ppl::nn::pmx::RotaryPositionEmbeddingParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "theta", &param->theta, -1)) {
        LOG(ERROR) << node->GetName() << ": missing theta";
        return RC_INVALID_VALUE;
    }

    onnx::utils::GetNodeAttr(pb_node, "bypass_key", &param->bypass_key, 0);
    onnx::utils::GetNodeAttr(pb_node, "rotary_dim", &param->rotary_dim, 0);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx