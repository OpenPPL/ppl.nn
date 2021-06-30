#include "ppl/nn/models/onnx/parsers/parse_pad_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParsePadParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::PadParam*>(arg);
    std::string mode = utils::GetNodeAttrByKey<std::string>(pb_node, "mode", "constant");
    if (mode == "constant") {
        param->mode = ppl::nn::common::PadParam::PAD_MODE_CONSTANT;
    } else if (mode == "reflect") {
        param->mode = ppl::nn::common::PadParam::PAD_MODE_REFLECT;
    } else if (mode == "edge") {
        param->mode = ppl::nn::common::PadParam::PAD_MODE_EDGE;
    } else {
        LOG(ERROR) << "Invalid pad mode " << mode << ".";
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
