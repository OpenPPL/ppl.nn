#include "ppl/nn/engines/common/onnx/sequence_at_op.h"
#include "ppl/nn/engines/common/onnx/sequence_at_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

KernelImpl* SequenceAtOp::CreateKernelImpl() const {
    return new SequenceAtKernel(node_);
}

}}} // namespace ppl::nn::common
