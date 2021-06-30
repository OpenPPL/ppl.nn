#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_PAD_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_PAD_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

#define PAD_PARAM_MAX_DIM_SIZE 5

struct PadParam {
    typedef uint32_t pad_mode_t;
    enum { PAD_MODE_CONSTANT = 0, PAD_MODE_REFLECT = 1, PAD_MODE_EDGE = 2 };

    pad_mode_t mode = PAD_MODE_CONSTANT;

    bool operator==(const PadParam& p) const {
        return this->mode == p.mode;
    }
};

}}} // namespace ppl::nn::common

#endif
