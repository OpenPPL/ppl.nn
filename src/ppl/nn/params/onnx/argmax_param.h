#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_ARGMAX_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_ARGMAX_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct ArgMaxParam {
    int32_t axis;
    int32_t keepdims;

    bool operator==(const ArgMaxParam& p) const {
        return this->axis == p.axis && this->keepdims == p.keepdims;
    }
};

}}} // namespace ppl::nn::common

#endif
