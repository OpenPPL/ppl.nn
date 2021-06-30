#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_LEAKY_RELU_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_LEAKY_RELU_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct LeakyReLUParam {
    float alpha;

    bool operator==(const LeakyReLUParam& p) const {
        return this->alpha == p.alpha;
    }
};

}}} // namespace ppl::nn::common

#endif
