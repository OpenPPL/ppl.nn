#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_SOFTMAX_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_SOFTMAX_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct SoftmaxParam {
    int32_t axis;

    bool operator==(const SoftmaxParam& p) const {
        return this->axis == p.axis;
    }
};

}}} // namespace ppl::nn::common

#endif
