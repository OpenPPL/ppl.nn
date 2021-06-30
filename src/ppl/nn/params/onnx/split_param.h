#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_SPLIT_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_SPLIT_PARAM_H_

#include <vector>
#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct SplitParam {
    std::vector<int32_t> split_point;
    int32_t axis;

    bool operator==(const SplitParam& p) const {
        return this->split_point == p.split_point && this->axis == p.axis;
    }
};

}}} // namespace ppl::nn::common

#endif
