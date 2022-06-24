// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/pad/pad_fp32.h"
#include "ppl/kernel/x86/fp32/reorder.h"
#include <string.h>

namespace ppl { namespace kernel { namespace x86 {

const int64_t c_blk = 16;

static inline void copy_16c(float *dst, const float *src)
{
    dst[0]  = src[0];
    dst[1]  = src[1];
    dst[2]  = src[2];
    dst[3]  = src[3];
    dst[4]  = src[4];
    dst[5]  = src[5];
    dst[6]  = src[6];
    dst[7]  = src[7];
    dst[8]  = src[8];
    dst[9]  = src[9];
    dst[10] = src[10];
    dst[11] = src[11];
    dst[12] = src[12];
    dst[13] = src[13];
    dst[14] = src[14];
    dst[15] = src[15];
}

template <pad_mode_type_t _mode>
inline void pad_n16cx_last_dim_pad_begin_fp32(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst);

template <>
inline void pad_n16cx_last_dim_pad_begin_fp32<PAD_MODE_CONSTANT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin * c_blk; i++) {
        dst[i] = constant_value;
    }
}

template <>
inline void pad_n16cx_last_dim_pad_begin_fp32<PAD_MODE_REFLECT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin; i++) {
        copy_16c(dst + i * c_blk, src + get_reflect_idx(i - pad_begin, input_length) * c_blk);
    }
}

template <>
inline void pad_n16cx_last_dim_pad_begin_fp32<PAD_MODE_EDGE>(
    const float *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_begin; i++) {
        copy_16c(dst + i * c_blk, src);
    }
}

template <pad_mode_type_t _mode>
inline void pad_n16cx_last_dim_pad_end_fp32(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst);

template <>
inline void pad_n16cx_last_dim_pad_end_fp32<PAD_MODE_CONSTANT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_end * c_blk; i++) {
        dst[i + (pad_begin + input_length) * c_blk] = constant_value;
    }
}

template <>
inline void pad_n16cx_last_dim_pad_end_fp32<PAD_MODE_REFLECT>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_end; i++) {
        copy_16c(dst + (i + pad_begin + input_length) * c_blk, src + get_reflect_idx(input_length + i, input_length) * c_blk);
    }
}

template <>
inline void pad_n16cx_last_dim_pad_end_fp32<PAD_MODE_EDGE>(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const float constant_value,
    float *dst)
{
    for (int64_t i = 0; i < pad_end; i++) {
        copy_16c(dst + (i + pad_begin + input_length) * c_blk, src + (input_length - 1) * c_blk);
    }
}

template <pad_mode_type_t _mode>
inline void pad_n16cx_begin_end_fp32(
    const float *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const int64_t stride_out,
    const float constant_value,
    float *dst);

template <>
inline void pad_n16cx_begin_end_fp32<PAD_MODE_CONSTANT>(
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
inline void pad_n16cx_begin_end_fp32<PAD_MODE_REFLECT>(
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
inline void pad_n16cx_begin_end_fp32<PAD_MODE_EDGE>(
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
ppl::common::RetCode pad_n16cx_recursive_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const int64_t *stride_in,
    const int64_t *stride_out,
    float constant_value,
    const int64_t dim_idx,
    const int64_t c_dim_idx,
    const bool has_paralleled,
    float *dst)
{
    const int64_t pad_begin = start_pads[dim_idx];
    const int64_t pad_end   = end_pads[dim_idx];

    const int64_t length    = dim_idx == c_dim_idx ? div_up(src_shape->GetDim(dim_idx), c_blk) : src_shape->GetDim(dim_idx);
    const int64_t dim_count = src_shape->GetDimCount();

    if (dim_idx == dim_count - 1) { // last dim
        pad_n16cx_last_dim_pad_begin_fp32<_mode>(src, pad_begin, length, constant_value, dst);
        memcpy(dst + pad_begin * c_blk, src, length * c_blk * sizeof(float));
        pad_n16cx_last_dim_pad_end_fp32<_mode>(src, pad_begin, pad_end, length, constant_value, dst);
    } else {
        if (length > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                pad_n16cx_recursive_fp32<_mode>(
                    src_shape,
                    dst_shape,
                    src + i * stride_in[dim_idx],
                    start_pads,
                    end_pads,
                    stride_in,
                    stride_out,
                    constant_value,
                    dim_idx + 1,
                    c_dim_idx,
                    true,
                    dst + (i + pad_begin) * stride_out[dim_idx]);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                pad_n16cx_recursive_fp32<_mode>(
                    src_shape,
                    dst_shape,
                    src + i * stride_in[dim_idx],
                    start_pads,
                    end_pads,
                    stride_in,
                    stride_out,
                    constant_value,
                    dim_idx + 1,
                    c_dim_idx,
                    has_paralleled,
                    dst + (i + pad_begin) * stride_out[dim_idx]);
            }
        }
        pad_n16cx_begin_end_fp32<_mode>(src, pad_begin, pad_end, length, stride_out[dim_idx], constant_value, dst);
    }

    return ppl::common::RC_SUCCESS;
}

template <pad_mode_type_t _mode>
ppl::common::RetCode pad_n16cx_fp32(
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

    const int64_t c_dim_idx = 1;
    if (start_pads[c_dim_idx] != 0 || end_pads[c_dim_idx] != 0) { // if pad on c, trans to n16cx to pad
        // if (true) {
        std::vector<float> temp_in(src_shape->CalcElementsExcludingPadding());
        std::vector<float> temp_out(dst_shape->CalcElementsExcludingPadding());
        auto src_shape_ndarray = *src_shape;
        auto dst_shape_ndarray = *dst_shape;
        src_shape_ndarray.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        dst_shape_ndarray.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        src_shape_ndarray.CalcPadding();
        dst_shape_ndarray.CalcPadding();

        ppl::common::RetCode ret = ppl::common::RC_SUCCESS;

        ret = reorder_n16cx_ndarray_fp32_avx(src_shape, src, (float*)temp_in.data());
        if (ret != ppl::common::RC_SUCCESS) {
            return ret;
        }

        ret = pad_ndarray_fp32<_mode>(&src_shape_ndarray, &dst_shape_ndarray, (float*)temp_in.data(), start_pads, end_pads, constant_value, temp_out.data());
        if (ret != ppl::common::RC_SUCCESS) {
            return ret;
        }

        ret = reorder_ndarray_n16cx_fp32_avx(&dst_shape_ndarray, (float*)temp_out.data(), dst);
        return ret;
    }

    int64_t stride_in[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t stride_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    stride_in[dim_count - 1]  = c_blk;
    stride_out[dim_count - 1] = c_blk;
    for (int64_t i = dim_count - 2; i >= 0; i--) {
        int64_t in_dim  = src_shape->GetDim(i + 1);
        int64_t out_dim = dst_shape->GetDim(i + 1);
        if (i + 1 == c_dim_idx) {
            in_dim  = div_up(in_dim, c_blk);
            out_dim = div_up(out_dim, c_blk);
        }
        stride_in[i]  = stride_in[i + 1] * in_dim;
        stride_out[i] = stride_out[i + 1] * out_dim;
    }

    return pad_n16cx_recursive_fp32<_mode>(
        src_shape,
        dst_shape,
        src,
        start_pads,
        end_pads,
        stride_in,
        stride_out,
        constant_value,
        0,
        c_dim_idx,
        false,
        dst);
}

template ppl::common::RetCode pad_n16cx_fp32<PAD_MODE_CONSTANT>(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

template ppl::common::RetCode pad_n16cx_fp32<PAD_MODE_REFLECT>(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

template ppl::common::RetCode pad_n16cx_fp32<PAD_MODE_EDGE>(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst);

ppl::common::RetCode pad_n16cx_constant_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float constant_value,
    float *dst)
{
    return pad_n16cx_fp32<PAD_MODE_CONSTANT>(src_shape, dst_shape, src, start_pads, end_pads, constant_value, dst);
}

ppl::common::RetCode pad_n16cx_reflect_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst)
{
    return pad_n16cx_fp32<PAD_MODE_REFLECT>(src_shape, dst_shape, src, start_pads, end_pads, 0, dst);
}

ppl::common::RetCode pad_n16cx_edge_fp32(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    float *dst)
{
    return pad_n16cx_fp32<PAD_MODE_EDGE>(src_shape, dst_shape, src, start_pads, end_pads, 0, dst);
}

}}}; // namespace ppl::kernel::x86
