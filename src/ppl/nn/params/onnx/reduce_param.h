#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_REDUCE_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_REDUCE_PARAM_H_

#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct ReduceParam {
    std::vector<int32_t> axes;
    bool keep_dims;

    bool operator==(const ReduceParam& p) const {
        return this->axes == p.axes && this->keep_dims == p.keep_dims;
    }
};

}}} // namespace ppl::nn::common

#endif
