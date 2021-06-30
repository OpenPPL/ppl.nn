#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_SCATTER_ELEMENTS_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_SCATTER_ELEMENTS_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct ScatterElementsParam {
    int32_t axis;

    bool operator==(const ScatterElementsParam& p) const {
        return this->axis == p.axis;
    }
};

}}} // namespace ppl::nn::common

#endif
