#include "ppl/nn/models/onnx/parsers/parse_leaky_relu_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseLeakyReLUParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::LeakyReLUParam*>(arg);
    param->alpha = utils::GetNodeAttrByKey<float>(pb_node, "alpha", 0.01);
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
