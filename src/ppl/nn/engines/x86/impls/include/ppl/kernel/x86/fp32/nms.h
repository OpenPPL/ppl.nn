#ifndef __ST_PPL_KERNEL_X86_FP32_NMS_H_
#define __ST_PPL_KERNEL_X86_FP32_NMS_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode nms_ndarray_fp32(
    const float *boxes,
    const float *scommons,
    const uint32_t num_boxes_in,
    const uint32_t batch,
    const uint32_t num_classes,
    const bool center_point_box,
    const int64_t max_output_boxes_per_batch_per_class,
    const float iou_threshold,
    const float scommon_threshold,
    int64_t *dst,
    int64_t *num_boxes_out);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_NMS_H_
