#include "ppl/nn/models/onnx/parsers/parse_convtranspose_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseConvTransposeParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ConvTransposeParam*>(arg);

    param->auto_pad = utils::GetNodeAttrByKey<std::string>(pb_node, "auto_pad", "");
    param->dilations = utils::GetNodeAttrsByKey<int32_t>(pb_node, "dilations");
    param->kernel_shape = utils::GetNodeAttrsByKey<int32_t>(pb_node, "kernel_shape");
    param->output_padding = utils::GetNodeAttrsByKey<int32_t>(pb_node, "output_padding");
    param->output_shape = utils::GetNodeAttrsByKey<int32_t>(pb_node, "output_shape");
    param->pads = utils::GetNodeAttrsByKey<int32_t>(pb_node, "pads");
    param->strides = utils::GetNodeAttrsByKey<int32_t>(pb_node, "strides");
    param->group = utils::GetNodeAttrByKey<int64_t>(pb_node, "group", 1);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
