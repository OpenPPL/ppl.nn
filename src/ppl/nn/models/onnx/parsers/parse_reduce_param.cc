#include "ppl/nn/models/onnx/parsers/parse_reduce_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseReduceParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ReduceParam*>(arg);

    param->axes = utils::GetNodeAttrsByKey<int32_t>(pb_node, "axes");
    int keepdims = utils::GetNodeAttrByKey<int>(pb_node, "keepdims", 1);
    param->keep_dims = (keepdims != 0);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
