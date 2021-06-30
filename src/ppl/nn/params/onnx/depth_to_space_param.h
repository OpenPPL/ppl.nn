#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_DEPTH_TO_SPACE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_DEPTH_TO_SPACE_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct DepthToSpaceParam {
    enum { DCR = 0, CRD = 1 };

    int32_t blocksize;
    int32_t mode;

    bool operator==(const DepthToSpaceParam& p) const {
        return this->blocksize == p.blocksize && this->mode == p.mode;
    }
};

}}} // namespace ppl::nn::common

#endif
