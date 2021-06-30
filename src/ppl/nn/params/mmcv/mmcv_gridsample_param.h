#ifndef _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_GRIDSAMPLE_PARAM_H
#define _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_GRIDSAMPLE_PARAM_H

#include <stdint.h>
#include <string>

namespace ppl { namespace nn { namespace common {

struct MMCVGridSampleParam {
    int64_t align_corners;
    int64_t interpolation_mode;
    int64_t padding_mode;

    bool operator==(const MMCVGridSampleParam& p) const {
        return this->align_corners == p.align_corners && this->interpolation_mode == p.interpolation_mode &&
            this->padding_mode == p.padding_mode;
    }
};

}}} // namespace ppl::nn::common

#endif
