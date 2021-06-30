#include "ppl/nn/oputils/onnx/reshape_depth_to_space.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeDepthToSpace(InputOutputInfo* info, const void* arg) {
    auto param = (const DepthToSpaceParam*)arg;
    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    auto out_shape0 = &info->GetOutput<TensorImpl>(0)->GetShape();
    out_shape0->Reshape({in_shape0.GetDim(0), in_shape0.GetDim(1) / (param->blocksize * param->blocksize),
                         in_shape0.GetDim(2) * param->blocksize, in_shape0.GetDim(3) * param->blocksize});
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
