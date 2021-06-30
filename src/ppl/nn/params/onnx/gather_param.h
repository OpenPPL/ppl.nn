#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_GATHER_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_GATHER_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct GatherParam {
    int32_t axis;

    bool operator==(const GatherParam& p) const {
        return this->axis == p.axis;
    }
};

}}} // namespace ppl::nn::common

#endif
