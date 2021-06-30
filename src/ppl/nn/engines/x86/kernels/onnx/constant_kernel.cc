#include "ppl/nn/engines/x86/kernels/onnx/constant_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode ConstantKernel::DoExecute(KernelExecContext* ctx) {
    auto output = ctx->GetOutput<TensorImpl>(0);
    GetDevice()->CopyFromHost(&output->GetBufferDesc(), param_->data.data(), output->GetShape());
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
