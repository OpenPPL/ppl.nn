#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_FLATTEN_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_FLATTEN_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct FlattenParam {
    int32_t axis;
    int32_t end_axis;

    bool operator==(const FlattenParam& p) const {
        return this->axis == p.axis && this->end_axis == p.end_axis;
    }
};

}}} // namespace ppl::nn::common

#endif
