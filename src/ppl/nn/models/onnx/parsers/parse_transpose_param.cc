#include "ppl/nn/models/onnx/parsers/parse_transpose_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseTransposeParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::TransposeParam*>(arg);
    param->perm = utils::GetNodeAttrsByKey<int32_t>(pb_node, "perm");
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
