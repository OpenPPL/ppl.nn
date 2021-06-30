#include "ppl/nn/models/onnx/parsers/parse_constant_of_shape_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"

namespace ppl { namespace nn { namespace onnx {

ppl::common::RetCode ParseConstantOfShapeParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::nn::common::ConstantOfShapeParam*>(arg);

    const ::onnx::TensorProto* value = utils::GetTensorProtoByKey(pb_node, "value");
    if (value == nullptr) {
        float f = 0.0;
        param->data_type = ppl::common::DATATYPE_FLOAT32;
        param->dims.push_back(1);
        param->data.assign((const char*)&f, sizeof(f));
    } else {
        ir::Shape shape;
        auto status = utils::ParseTensorProto(*value, &param->data, &shape);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "parse attribute of node[" << pb_node.name()
                       << "] failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }

        param->data_type = shape.data_type;
        param->dims = std::move(shape.dims);
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
