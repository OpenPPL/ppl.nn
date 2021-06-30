#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaOptKernel::SetCommonParam(const OptKernelOptions& options) {
    auto node = GetNode();

    common_param_.output_tensor_info.resize(node->GetOutputCount());
    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto iter = options.tensors->find(edge_id);
        if (iter == options.tensors->end()) {
            LOG(ERROR) << "can not find edge " << edge_id;
            return RC_NOT_FOUND;
        }
        common_param_.output_tensor_info[i].data_format = iter->second->GetShape().GetDataFormat();
        common_param_.output_tensor_info[i].data_type = iter->second->GetShape().GetDataType();
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
