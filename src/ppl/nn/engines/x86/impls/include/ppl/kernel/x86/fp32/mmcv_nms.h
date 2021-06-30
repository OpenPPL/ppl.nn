#ifndef __ST_PPL_KERNEL_X86_FP32_MMCV_NMS_H_
#define __ST_PPL_KERNEL_X86_FP32_MMCV_NMS_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode mmcv_nms_ndarray_fp32(
        const float *boxes,
        const float *scores,
        const uint32_t num_boxes_in,
        const float iou_threshold,
        const int64_t offset,
        int64_t *dst,
        int64_t *num_boxes_out);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_MMCV_NMS_H_
