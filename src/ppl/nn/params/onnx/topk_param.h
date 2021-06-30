#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_TOPK_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_TOPK_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct TopKParam {
    int32_t axis;
    int32_t largest;
    int32_t sorted;

    bool operator==(const TopKParam& p) const {
        return this->axis == p.axis && this->largest == p.largest && this->sorted == p.sorted;
    }
};

}}} // namespace ppl::nn::common

#endif