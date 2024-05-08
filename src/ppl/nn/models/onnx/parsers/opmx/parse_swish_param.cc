#include "ppl/nn/models/onnx/parsers/opmx/parse_swish_param.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace opmx {

RetCode ParseSwishParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node*,
                       ir::Attr* arg) {
    auto param = static_cast<ppl::nn::opmx::SwishParam*>(arg);
    onnx::utils::GetNodeAttr(pb_node, "beta", &param->beta, 1.0f);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx