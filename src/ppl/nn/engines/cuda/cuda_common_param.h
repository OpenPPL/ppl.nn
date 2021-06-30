#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_COMMON_PARAM_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_COMMON_PARAM_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "ppl/common/types.h"
#include "ppl/nn/common/types.h"

namespace ppl { namespace nn { namespace cuda {

struct OutputTensorInfo {
    ppl::common::datatype_t data_type;
    ppl::common::dataformat_t data_format;
};

struct CudaCommonParam {
    std::vector<OutputTensorInfo> output_tensor_info;
    ppl::common::datatype_t kernel_default_type;
};

}}} // namespace ppl::nn::cuda

#endif
