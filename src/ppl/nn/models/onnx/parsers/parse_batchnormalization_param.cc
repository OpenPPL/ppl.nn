#include "ppl/nn/models/onnx/parsers/parse_batchnormalization_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseBatchNormalizationParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*,
                                                  ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::BatchNormalizationParam*>(arg);
    param->epsilon = utils::GetNodeAttrByKey<float>(pb_node, "epsilon", 1e-5);
    param->momentum = utils::GetNodeAttrByKey<float>(pb_node, "momentum", 0.9);
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
