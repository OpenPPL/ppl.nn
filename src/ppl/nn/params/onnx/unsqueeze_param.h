#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_UNSQUEEZE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_UNSQUEEZE_PARAM_H_

#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct UnsqueezeParam {
    std::vector<int32_t> axes;

    bool operator==(const UnsqueezeParam& p) const {
        return this->axes == p.axes;
    }
};

}}} // namespace ppl::nn::common

#endif
