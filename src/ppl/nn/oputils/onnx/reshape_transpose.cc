#include "ppl/nn/oputils/onnx/reshape_transpose.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeTranspose(InputOutputInfo* info, const void* arg) {
    auto param = (const TransposeParam*)arg;
    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();

    auto modified_perm = param->perm;
    if (modified_perm.empty()) { // perm is empty, default is reverse dimention.
        auto dim_count = info->GetInput<TensorImpl>(0)->GetShape().GetDimCount();
        modified_perm.resize(dim_count);
        for (size_t i = 0; i < dim_count; ++i) {
            modified_perm[i] = dim_count - i - 1;
        }
    }

    std::vector<int64_t> out_dims(in_shape0.GetDimCount());
    for (uint32_t i = 0; i < in_shape0.GetDimCount(); ++i) {
        out_dims[i] = in_shape0.GetDim(modified_perm[i]);
    }

    info->GetOutput<TensorImpl>(0)->GetShape().Reshape(out_dims);
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
