#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_CAST_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_CAST_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct CastParam {
    int32_t to;

    bool operator==(const CastParam& p) const {
        return this->to == p.to;
    }
};

}}} // namespace ppl::nn::common

#endif
