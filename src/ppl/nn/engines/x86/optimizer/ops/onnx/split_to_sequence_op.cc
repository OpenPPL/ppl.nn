#include "ppl/nn/engines/x86/optimizer/ops/onnx/split_to_sequence_op.h"
#include "ppl/nn/common/logger.h"
#include <cstring>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SplitToSequenceOp::Init(const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph_data;
    auto attr_ref = graph_data->attrs.find(node->GetId());
    if (attr_ref == graph_data->attrs.end()) {
        LOG(ERROR) << "cannot find attr for SplitToSequenceOp[" << node->GetName() << "]";
        return RC_NOT_FOUND;
    }

    auto param = static_cast<ppl::nn::onnx::SplitToSequenceParam*>(attr_ref->second.get());
    op_.Init(param->axis, param->keepdims, common::SplitToSequenceOp::GenericSplitFunc);
    return RC_SUCCESS;
}

KernelImpl* SplitToSequenceOp::CreateKernelImpl() const {
    return op_.CreateKernelImpl();
}

}}} // namespace ppl::nn::x86
