#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_NON_MAX_SUPRESSION_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_NON_MAX_SUPRESSION_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct NonMaxSuppressionParam {
    int32_t center_point_box;

    bool operator==(const NonMaxSuppressionParam& p) const {
        return this->center_point_box == p.center_point_box;
    }
};

}}} // namespace ppl::nn::common

#endif
