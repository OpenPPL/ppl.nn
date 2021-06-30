#include "ppl/nn/models/onnx/parsers/parse_split_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseSplitParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::SplitParam*>(arg);
    param->axis = utils::GetNodeAttrByKey(pb_node, "axis", 0);
    param->split_point = utils::GetNodeAttrsByKey<int32_t>(pb_node, "split");
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
