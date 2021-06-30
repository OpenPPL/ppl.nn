#include "ppl/nn/engines/x86/optimizer/ops/onnx/if_op.h"
#include "ppl/nn/engines/common/onnx/if_kernel.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode IfOp::Init(const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto attr_ref = graph_data->attrs.find(node->GetId());
    if (attr_ref == graph_data->attrs.end()) {
        LOG(ERROR) << "cannot find attr for if kernel[" << node->GetName() << "]";
        return RC_NOT_FOUND;
    }

    auto if_param = static_cast<ppl::nn::onnx::IfParam*>(attr_ref->second.get());
    return op_.Init(options.resource, if_param);
}

KernelImpl* IfOp::CreateKernelImpl() const {
    return op_.CreateKernelImpl();
}

}}} // namespace ppl::nn::x86
