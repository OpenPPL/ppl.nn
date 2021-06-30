#ifndef __ST_PPL_KERNEL_X86_FP32_PAD_PAD_FP32_H_
#define __ST_PPL_KERNEL_X86_FP32_PAD_PAD_FP32_H_

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

enum pad_mode_type_t {
    PAD_MODE_CONSTANT = 0,
    PAD_MODE_REFLECT  = 1,
    PAD_MODE_EDGE     = 2
};

static inline int32_t get_reflect_idx(
    int32_t idx,
    int32_t length)
{
    while (idx < 0 || idx >= length) {
        if (idx < 0) {
            idx = -idx;
        }
        if (idx >= length) {
            idx = 2 * length - 2 - idx;
        }
    }
    return idx;
}

template <pad_mode_type_t _mode>
ppl::common::RetCode pad_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

template <pad_mode_type_t _mode>
ppl::common::RetCode pad_n16cx_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

}}}; // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_PAD_PAD_FP32_H_