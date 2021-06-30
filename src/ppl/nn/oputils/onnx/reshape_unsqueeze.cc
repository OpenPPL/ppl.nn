#include "ppl/nn/oputils/onnx/reshape_unsqueeze.h"
#include "ppl/nn/common/logger.h"
#include <algorithm>
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeUnsqueeze(InputOutputInfo* info, const void* arg) {
    auto param = (const UnsqueezeParam*)arg;
    std::vector<int32_t> axes(param->axes.size());

    const TensorShape& input = info->GetInput<TensorImpl>(0)->GetShape();
    const int32_t out_dim_count = (int32_t)input.GetRealDimCount() + param->axes.size();

    for (uint32_t i = 0; i < param->axes.size(); ++i) {
        if (param->axes[i] < (int32_t)(-out_dim_count) || param->axes[i] >= (int32_t)out_dim_count) {
            LOG(ERROR) << "axes overflow.";
            return RC_INVALID_VALUE;
        }
        if (param->axes[i] < 0) {
            axes[i] = out_dim_count + param->axes[i];
        } else {
            axes[i] = param->axes[i];
        }
    }

    std::sort(axes.begin(), axes.end());
    std::vector<int64_t> output_dim(out_dim_count);
    for (int32_t oid = 0, aid = 0, iid = 0; oid < out_dim_count; ++oid) {
        if (aid < (int32_t)axes.size() && oid == axes[aid]) {
            output_dim[oid] = 1;
            ++aid;
        } else {
            output_dim[oid] = input.GetDim(iid);
            ++iid;
        }
    }

    info->GetOutput<TensorImpl>(0)->GetShape().Reshape(output_dim);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
