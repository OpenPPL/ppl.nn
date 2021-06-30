#include "ppl/nn/models/onnx/parsers/parse_mmcv_gridsample_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseMMCVGridSampleParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::MMCVGridSampleParam*>(arg);

    param->align_corners = utils::GetNodeAttrByKey<int64_t>(pb_node, "align_corners", 0);
    param->interpolation_mode = utils::GetNodeAttrByKey<int64_t>(pb_node, "interpolation_mode", 0);
    param->padding_mode = utils::GetNodeAttrByKey<int64_t>(pb_node, "padding_mode", 0);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
