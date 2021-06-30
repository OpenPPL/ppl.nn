#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_TRANSPOSE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_TRANSPOSE_PARAM_H_

#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct TransposeParam {
    std::vector<int32_t> perm;
    bool reverse = false;

    bool operator==(const TransposeParam& p) const {
        return this->perm == p.perm && this->reverse == p.reverse;
    }
};

}}} // namespace ppl::nn::common

#endif
