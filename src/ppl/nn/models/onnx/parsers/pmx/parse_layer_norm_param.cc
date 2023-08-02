#include "ppl/nn/models/onnx/parsers/pmx/parse_layer_norm_param.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ParseLayerNormParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node*,
                            ir::Attr* arg) {
    auto param = static_cast<ppl::nn::pmx::LayerNormParam*>(arg);
    int32_t elementwise_affine, skip_term;
    onnx::utils::GetNodeAttr(pb_node, "elementwise_affine", &elementwise_affine, 0);
    onnx::utils::GetNodeAttr(pb_node, "skip_term", &skip_term, 0);
    param->elementwise_affine = elementwise_affine;
    param->skip_term = skip_term;
    onnx::utils::GetNodeAttr(pb_node, "axis", &param->axis, -1);
    onnx::utils::GetNodeAttr(pb_node, "eps", &param->eps, 1e-5);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx