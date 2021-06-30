#include "ppl/nn/models/onnx/parsers/parse_topk_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {
ppl::common::RetCode ParseTopKParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::TopKParam*>(arg);

    param->axis = utils::GetNodeAttrByKey<int32_t>(pb_node, "axis", -1);
    param->largest = utils::GetNodeAttrByKey<int32_t>(pb_node, "largest", 1);
    param->sorted = utils::GetNodeAttrByKey<int32_t>(pb_node, "sorted", 1);

    return ppl::common::RC_SUCCESS;
}
}}} // namespace ppl::nn::onnx
