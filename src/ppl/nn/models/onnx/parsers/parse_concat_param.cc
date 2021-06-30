#include "ppl/nn/models/onnx/parsers/parse_concat_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseConcatParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ConcatParam*>(arg);

    int32_t axis = utils::GetNodeAttrByKey<int32_t>(pb_node, "axis", INT32_MAX);
    if (axis == INT32_MAX) {
        LOG(ERROR) << "axis is required.";
        return ppl::common::RC_INVALID_VALUE;
    }

    param->axis = axis;

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
