#include "ppl/nn/models/onnx/parsers/parse_maxunpool_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseMaxUnpoolParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::MaxUnpoolParam*>(arg);

    param->kernel_shape = utils::GetNodeAttrsByKey<int32_t>(pb_node, "kernel_shape");
    param->strides = utils::GetNodeAttrsByKey<int32_t>(pb_node, "strides");
    param->pads = utils::GetNodeAttrsByKey<int32_t>(pb_node, "pads");

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
