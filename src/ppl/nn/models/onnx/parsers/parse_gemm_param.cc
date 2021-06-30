#include "ppl/nn/models/onnx/parsers/parse_gemm_param.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseGemmParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::GemmParam*>(arg);

    param->alpha = utils::GetNodeAttrByKey<float>(pb_node, "alpha", 1.0f);
    param->beta = utils::GetNodeAttrByKey<float>(pb_node, "beta", 1.0f);
    param->transA = utils::GetNodeAttrByKey<int32_t>(pb_node, "transA", 0);
    param->transB = utils::GetNodeAttrByKey<int32_t>(pb_node, "transB", 0);
    param->N = 0; // set by opcontext

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
