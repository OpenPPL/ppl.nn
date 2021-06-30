#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_POOLING_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_POOLING_PARAM_H_

#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace common {

struct PoolingParam {
    typedef enum { POOLING_MAX = 0, POOLING_AVERAGE_EXCLUDE = 1, POOLING_AVERAGE_INCLUDE = 2 } pooling_mode_t;

    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> dilations;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    int32_t mode;
    int32_t ceil_mode;
    int32_t global_pooling;

    bool operator==(const PoolingParam& p) const {
        return this->kernel_shape == p.kernel_shape && this->dilations == p.dilations && this->strides == p.strides &&
            this->pads == p.pads && this->mode == p.mode && this->ceil_mode == p.ceil_mode &&
            this->global_pooling == p.global_pooling;
    }
};

}}} // namespace ppl::nn::common

#endif
