#include "ppl/nn/models/onnx/parsers/parse_non_max_suppression_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseNonMaxSuppressionParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*,
                                                 ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::NonMaxSuppressionParam*>(arg);
    param->center_point_box = utils::GetNodeAttrByKey<int>(pb_node, "center_point_box", 0);
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
