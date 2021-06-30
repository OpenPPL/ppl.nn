#include "ppl/nn/models/onnx/parsers/parse_gather_nd_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseGatherNDParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::GatherNDParam*>(arg);
    param->batch_dims = utils::GetNodeAttrByKey<int32_t>(pb_node, "batch_dims", 0);
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
