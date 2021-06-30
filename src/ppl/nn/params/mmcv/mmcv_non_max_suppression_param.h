#ifndef _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_NONMAXSUPPRESSION_PARAM_H_
#define _ST_HPC_PPL_NN_PARAMS_MMCV_MMCV_NONMAXSUPPRESSION_PARAM_H_

#include <stdint.h>

namespace ppl { namespace nn { namespace common {

struct MMCVNMSParam {
    float iou_threshold;
    int64_t offset;

    bool operator==(const MMCVNMSParam& p) const {
        return this->iou_threshold == p.iou_threshold && this->offset == p.offset;
    }
};

}}} // namespace ppl::nn::common

#endif
