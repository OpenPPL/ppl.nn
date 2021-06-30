#include "ppl/nn/models/onnx/parsers/parse_unsqueeze_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseUnsqueezeParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::UnsqueezeParam*>(arg);

    param->axes = utils::GetNodeAttrsByKey<int32_t>(pb_node, "axes");
    if (param->axes.empty()) {
        LOG(ERROR) << "axes is required.";
        return ppl::common::RC_INVALID_VALUE;
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
