#include "ppl/nn/models/onnx/parsers/parse_gather_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseGatherParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::GatherParam*>(arg);
    param->axis = utils::GetNodeAttrByKey<int32_t>(pb_node, "axis", 0);
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
