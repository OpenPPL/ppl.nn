#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_GATHER_ND_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_GATHER_ND_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct GatherNDParam {
    int32_t batch_dims; // this attribute not included in opset=11, so will not process this
                        // attribute now

    bool operator==(const GatherNDParam& p) const {
        return this->batch_dims == p.batch_dims;
    }
};

}}} // namespace ppl::nn::common

#endif
