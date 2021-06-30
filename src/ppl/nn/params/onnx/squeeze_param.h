#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_SQUEEZE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_SQUEEZE_PARAM_H_

#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct SqueezeParam {
    std::vector<int32_t> axes;

    bool operator==(const SqueezeParam& p) const {
        return this->axes == p.axes;
    }
};

}}} // namespace ppl::nn::common

#endif
