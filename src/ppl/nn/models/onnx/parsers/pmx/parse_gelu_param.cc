#include "ppl/nn/models/onnx/parsers/pmx/parse_gelu_param.h"
#include "ppl/nn/models/onnx/utils.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ParseGELUParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node*,
                       ir::Attr* arg) {
    auto param = static_cast<ppl::nn::pmx::GELUParam*>(arg);
    int32_t approximate;
    onnx::utils::GetNodeAttr(pb_node, "approximate", &approximate, 0);
    param->approximate = approximate;
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx