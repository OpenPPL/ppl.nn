#include "ppl/nn/models/onnx/parsers/parse_mmcv_roialign_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseMMCVROIAlignParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::MMCVROIAlignParam*>(arg);

    param->aligned = utils::GetNodeAttrByKey<int64_t>(pb_node, "aligned", 0);
    param->aligned_height = utils::GetNodeAttrByKey<int64_t>(pb_node, "output_height", 0);
    param->aligned_width = utils::GetNodeAttrByKey<int64_t>(pb_node, "output_width", 0);
    param->pool_mode = utils::GetNodeAttrByKey<std::string>(pb_node, "mode", "avg");
    param->sampling_ratio = utils::GetNodeAttrByKey<float>(pb_node, "sampling_ratio", 0);
    param->spatial_scale = utils::GetNodeAttrByKey<float>(pb_node, "spatial_scale", 0);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
