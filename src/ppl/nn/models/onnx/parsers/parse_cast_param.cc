#include "ppl/nn/models/onnx/parsers/parse_cast_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseCastParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto cast_param = static_cast<ppl::nn::common::CastParam*>(arg);

    int32_t to = utils::GetNodeAttrByKey<int32_t>(pb_node, "to", 1);
    cast_param->to = utils::ConvertOnnxDataTypeToPplDataType(to);

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
