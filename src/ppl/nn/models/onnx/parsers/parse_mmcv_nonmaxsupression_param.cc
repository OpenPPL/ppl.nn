#include "ppl/nn/models/onnx/parsers/parse_mmcv_nonmaxsupression_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseMMCVNMSParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::MMCVNMSParam*>(arg);

    param->iou_threshold = utils::GetNodeAttrByKey<float>(pb_node, "iou_threshold", 0);
    param->offset = utils::GetNodeAttrByKey<int32_t>(pb_node, "offset", 0);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
