#ifndef _ST_HPC_PPL_NN_PARAMS_ONNX_ROIALIGN_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_ONNX_ROIALIGN_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct ROIAlignParam {
    enum { ONNXROIAlignMode_AVG = 0, ONNXROIAlignMode_MAX = 1 };

    int32_t mode;
    int32_t output_height;
    int32_t output_width;
    int32_t sampling_ratio;
    float spatial_scale;

    bool operator==(const ROIAlignParam& p) const {
        return this->mode == p.mode && this->output_height == p.output_height && this->output_width == p.output_width &&
            this->sampling_ratio == p.sampling_ratio && this->spatial_scale == p.spatial_scale;
    }
};

}}} // namespace ppl::nn::common

#endif
