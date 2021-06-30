#ifndef _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_ROIALIGN_PARAM_H
#define _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_ROIALIGN_PARAM_H

#include <stdint.h>
#include <string>

namespace ppl { namespace nn { namespace common {

struct MMCVROIAlignParam {
    int64_t aligned;
    int64_t aligned_height;
    int64_t aligned_width;
    std::string pool_mode;
    int64_t sampling_ratio;
    float spatial_scale;

    bool operator==(const MMCVROIAlignParam& p) const {
        return this->aligned == p.aligned && this->aligned_height == p.aligned_height &&
            this->aligned_width == p.aligned_width && this->pool_mode == p.pool_mode &&
            this->sampling_ratio == p.sampling_ratio && this->spatial_scale == p.spatial_scale;
    }
};

}}} // namespace ppl::nn::common

#endif
