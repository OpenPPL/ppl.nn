#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"
#include "ppl/common/sys.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

X86OptKernel::X86OptKernel(const ir::Node* node) : OptKernel(node) {
    common_param_.output_formats.resize(node->GetOutputCount(), DATAFORMAT_NDARRAY);
}

}}} // namespace ppl::nn::x86
