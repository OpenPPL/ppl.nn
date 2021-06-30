#include "ppl/nn/models/onnx/parsers/parse_constant_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

static ppl::common::RetCode DoParse(const ::onnx::TensorProto& pb_tensor, ppl::nn::common::ConstantParam* param) {
    ir::Shape shape;
    auto status = utils::ParseTensorProto(pb_tensor, &param->data, &shape);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "parse `value` failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }

    param->data_type = shape.data_type;
    param->data_format = shape.data_format;
    param->dims = std::move(shape.dims);

    return ppl::common::RC_SUCCESS;
}

static const char* g_keys[] = {
    "value", // The value for the elements of the output tensor.
    "value_float", // The value for the sole element for the scalar, float32, output tensor.
    "value_floats", // The values for the elements for the 1D, float32, output tensor.
    "value_int", // The value for the sole element for the scalar, int64, output tensor.
    "value_ints", // The values for the elements for the 1D, int64, output tensor.
    // "value_string", // The value for the sole element for the scalar, UTF-8 string, output tensor.
    // "value_strings", // The values for the elements for the 1D, UTF-8 string, output tensor.
    nullptr,
};

ppl::common::RetCode ParseConstantParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ConstantParam*>(arg);

    for (uint32_t i = 0; g_keys[i]; ++i) {
        const ::onnx::TensorProto* value = utils::GetTensorProtoByKey(pb_node, g_keys[i]);
        if (value) {
            return DoParse(*value, param);
        }
    }

    LOG(ERROR) << "cannot find supported fields in arg of Constant[" << pb_node.name() << "]";
    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::onnx
