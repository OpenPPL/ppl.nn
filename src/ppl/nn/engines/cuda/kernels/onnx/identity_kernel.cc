#include "ppl/nn/engines/cuda/kernels/onnx/identity_kernel.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode IdentityKernel::DoExecute(KernelExecContext* ctx) {
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
