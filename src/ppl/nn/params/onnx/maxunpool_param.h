#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_MAXUNPOOL_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_MAXUNPOOL_PARAM_H_

#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct MaxUnpoolParam {
    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> pads;
    std::vector<int32_t> strides;

    bool operator==(const MaxUnpoolParam& p) const {
        return this->kernel_shape == p.kernel_shape && this->pads == p.pads && this->strides == p.strides;
    }
};

}}} // namespace ppl::nn::common

#endif
