#include "ppl/nn/models/onnx/parsers/parse_split_to_sequence_param.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseSplitToSequenceParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<SplitToSequenceParam*>(arg);
    param->axis = utils::GetNodeAttrByKey<int32_t>(pb_node, "axis", 0);
    param->keepdims = utils::GetNodeAttrByKey<int32_t>(pb_node, "keepdims", 1);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
