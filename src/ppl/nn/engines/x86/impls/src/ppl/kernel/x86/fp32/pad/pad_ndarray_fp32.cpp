#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/pad/pad_fp32.h"
#include <string.h>

namespace ppl { namespace kernel { namespace x86 {

template <pad_mode_type_t _mode>
inline void pad_ndarray_last_dim_pad_begin_fp32(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst);

template <>
inline void pad_ndarray_last_dim_pad_begin_fp32<PAD_MODE_CONSTANT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin; i++) {
        dst[i] = constant_value;
    }
}

template <>
inline void pad_ndarray_last_dim_pad_begin_fp32<PAD_MODE_REFLECT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin; i++) {
        dst[i] = src[get_reflect_idx(i - pad_begin, input_length)];
    }
}

template <>
inline void pad_ndarray_last_dim_pad_begin_fp32<PAD_MODE_EDGE>(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin; i++) {
        dst[i] = src[0];
    }
}

template <pad_mode_type_t _mode>
inline void pad_ndarray_last_dim_pad_end_fp32(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst);

template <>
inline void pad_ndarray_last_dim_pad_end_fp32<PAD_MODE_CONSTANT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_end; i++) {
        dst[i + pad_begin + input_length] = constant_value;
    }
}

template <>
inline void pad_ndarray_last_dim_pad_end_fp32<PAD_MODE_REFLECT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_end; i++) {
        dst[i + pad_begin + input_length] = src[get_reflect_idx(input_length + i, input_length)];
    }
}

template <>
inline void pad_ndarray_last_dim_pad_end_fp32<PAD_MODE_EDGE>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_end; i++) {
        dst[i + pad_begin + input_length] = src[input_length - 1];
    }
}

template <pad_mode_type_t _mode>
inline void pad_ndarray_begin_end_fp32(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const int64_t stride_out,
    const float constant_value,
    float *dst);

template <>
inline void pad_ndarray_begin_end_fp32<PAD_MODE_CONSTANT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const int64_t stride_out,
    const float constant_value,
    float *dst)
{
    const int64_t pad_end_offset = (pad_begin + input_length) * stride_out;
    if (constant_value == 0.0f) {
        memset(dst, 0, pad_begin * stride_out * sizeof(float));
        memset(dst + pad_end_offset, 0, pad_end * stride_out * sizeof(float));
        return;
    }

    for (int64_t i = 0; i < pad_begin * stride_out; i++) {
        dst[i] = constant_value; // TODO: optimize here
    }

    for (int64_t i = 0; i < pad_end * stride_out; i++) {
        dst[i + pad_end_offset] = constant_value; // TODO: optimize here
    }
}

template <>
inline void pad_ndarray_begin_end_fp32<PAD_MODE_REFLECT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const int64_t stride_out,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin; i++) {
        int64_t copy_idx = get_reflect_idx(i - pad_begin, input_length) + pad_begin;
        memcpy(dst + i * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(float));
    }

    for (int64_t i = 0; i < pad_end; i++) {
        int64_t copy_idx      = get_reflect_idx(input_length + i, input_length) + pad_begin;
        const int64_t dst_idx = pad_begin + input_length + i;
        memcpy(dst + dst_idx * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(float));
    }
}

template <>
inline void pad_ndarray_begin_end_fp32<PAD_MODE_EDGE>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const int64_t stride_out,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin; i++) {
        int64_t copy_idx = pad_begin;
        memcpy(dst + i * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(float));
    }

    for (int64_t i = 0; i < pad_end; i++) {
        int64_t copy_idx      = pad_begin + input_length - 1;
        const int64_t dst_idx = pad_begin + input_length + i;
        memcpy(dst + dst_idx * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(float));
    }
}

template <pad_mode_type_t _mode>
ppl::common::RetCode pad_ndarray_recursive_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const int64_t *stride_in,
    const int64_t *stride_out,
    float constant_value,
    const int64_t dim_idx,
    const bool has_paralleled,
    float *dst)
{
    const int64_t pad_begin = start_pads[dim_idx];
    const int64_t pad_end   = end_pads[dim_idx];

    if (dim_idx == src_shape->GetDimCount() - 1) { // last dim
        pad_ndarray_last_dim_pad_begin_fp32<_mode>(src, pad_begin, src_shape->GetDim(dim_idx), constant_value, dst);
        memcpy(dst + pad_begin, src, src_shape->GetDim(dim_idx) * sizeof(float));
        pad_ndarray_last_dim_pad_end_fp32<_mode>(src, pad_begin, pad_end, src_shape->GetDim(dim_idx), constant_value, dst);
    } else {
        if (src_shape->GetDim(dim_idx) > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < src_shape->GetDim(dim_idx); i++) {
                pad_ndarray_recursive_fp32<_mode>(
                    src_shape,
                    dst_shape,
                    src + i * stride_in[dim_idx],
                    start_pads,
                    end_pads,
                    stride_in,
                    stride_out,
                    constant_value,
                    dim_idx + 1,
                    true,
                    dst + (i + pad_begin) * stride_out[dim_idx]);
            }
        } else {
            for (int64_t i = 0; i < src_shape->GetDim(dim_idx); i++) {
                pad_ndarray_recursive_fp32<_mode>(
                    src_shape,
                    dst_shape,
                    src + i * stride_in[dim_idx],
                    start_pads,
                    end_pads,
                    stride_in,
                    stride_out,
                    constant_value,
                    dim_idx + 1,
                    has_paralleled,
                    dst + (i + pad_begin) * stride_out[dim_idx]);
            }
        }
        pad_ndarray_begin_end_fp32<_mode>(src, pad_begin, pad_end, src_shape->GetDim(dim_idx), stride_out[dim_idx], constant_value, dst);
    }
    return ppl::common::RC_SUCCESS;
}

template <pad_mode_type_t _mode>
ppl::common::RetCode pad_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst)
{
    const int64_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t stride_in[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    stride_in[dim_count - 1]  = 1;
    stride_out[dim_count - 1] = 1;
    for (int64_t i = dst_shape->GetDimCount() - 2; i >= 0; i--) {
        stride_in[i]  = stride_in[i + 1] * src_shape->GetDim(i + 1);
        stride_out[i] = stride_out[i + 1] * dst_shape->GetDim(i + 1);
    }

    return pad_ndarray_recursive_fp32<_mode>(
        src_shape,
        dst_shape,
        src,
        start_pads,
        end_pads,
        stride_in,
        stride_out,
        constant_value,
        0,
        false,
        dst);
}

template ppl::common::RetCode pad_ndarray_fp32<PAD_MODE_CONSTANT>(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

template ppl::common::RetCode pad_ndarray_fp32<PAD_MODE_REFLECT>(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

template ppl::common::RetCode pad_ndarray_fp32<PAD_MODE_EDGE>(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

ppl::common::RetCode pad_ndarray_constant_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst)
{
    return pad_ndarray_fp32<PAD_MODE_CONSTANT>(src_shape, dst_shape, src, start_pads, end_pads, constant_value, dst);
}

ppl::common::RetCode pad_ndarray_reflect_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst)
{
    return pad_ndarray_fp32<PAD_MODE_REFLECT>(src_shape, dst_shape, src, start_pads, end_pads, 0, dst);
}

ppl::common::RetCode pad_ndarray_edge_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst)
{
    return pad_ndarray_fp32<PAD_MODE_EDGE>(src_shape, dst_shape, src, start_pads, end_pads, 0, dst);
}

}}}; // namespace ppl::kernel::x86
