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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_PAD_NEON_PAD_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_PAD_NEON_PAD_NBCX_COMMON_H_

#include <vector>
#include <string.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/pad/neon/pad_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, int32_t c_blk, pad_mode_type_t _mode>
inline void pad_nbcx_last_dim_pad_begin_common(
    const eT *src,
    const int64_t pad_begin,
    const int64_t input_length,
    const eT constant_value,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    if (_mode == PAD_MODE_CONSTANT) {
        const vecType v_constant = vdup_n<eT, eN>(constant_value);
        for (int64_t i = 0; i < pad_begin; i++) {
            vst<eT, eN>(dst + i * c_blk, v_constant);
        }
    } else if (_mode == PAD_MODE_REFLECT) {
        for (int64_t i = 0; i < pad_begin; i++) {
            const vecType v_data = vld<eT, eN>(src + get_reflect_idx(i - pad_begin, input_length) * c_blk);
            vst<eT, eN>(dst + i * c_blk, v_data);
        }
    } else if (_mode == PAD_MODE_EDGE) {
        for (int64_t i = 0; i < pad_begin; i++) {
            const vecType v_data = vld<eT, eN>(src);
            vst<eT, eN>(dst + i * c_blk, v_data);
        }
    }
}

template <typename eT, int32_t c_blk, pad_mode_type_t _mode>
inline void pad_nbcx_last_dim_pad_end_common(
    const eT *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const eT constant_value,
    eT *dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    if (_mode == PAD_MODE_CONSTANT) {
        const vecType v_constant = vdup_n<eT, eN>(constant_value);
        for (int64_t i = 0; i < pad_end; i++) {
            vst<eT, eN>(dst + (pad_begin + input_length + i) * c_blk, v_constant);
        }
    } else if (_mode == PAD_MODE_REFLECT) {
        for (int64_t i = 0; i < pad_end; i++) {
            const vecType v_data = vld<eT, eN>(src + get_reflect_idx(input_length + i, input_length) * c_blk);
            vst<eT, eN>(dst + (i + pad_begin + input_length) * c_blk, v_data);
        }
    } else if (_mode == PAD_MODE_EDGE) {
        for (int64_t i = 0; i < pad_end; i++) {
            const vecType v_data = vld<eT, eN>(src + (input_length - 1) * c_blk);
            vst<eT, eN>(dst + (i + pad_begin + input_length) * c_blk, v_data);
        }
    }
}

template <typename eT, int32_t c_blk, pad_mode_type_t _mode>
inline void pad_nbcx_begin_end_common(
    const eT *src,
    const int64_t pad_begin,
    const int64_t pad_end,
    const int64_t input_length,
    const int64_t stride_out,
    const eT constant_value,
    eT *dst)
{
    if (_mode == PAD_MODE_CONSTANT) {
        const int64_t pad_end_offset = (pad_begin + input_length) * stride_out;
        if (constant_value == (eT)0) {
            memset(dst, 0, pad_begin * stride_out * sizeof(eT));
            memset(dst + pad_end_offset, 0, pad_end * stride_out * sizeof(eT));
            return;
        }

        for (int64_t i = 0; i < pad_begin * stride_out; i++) {
            dst[i] = constant_value; // TODO: optimize here
        }

        for (int64_t i = 0; i < pad_end * stride_out; i++) {
            dst[i + pad_end_offset] = constant_value; // TODO: optimize here
        }
    } else if (_mode == PAD_MODE_REFLECT) {
        for (int64_t i = 0; i < pad_begin; i++) {
            int64_t copy_idx = get_reflect_idx(i - pad_begin, input_length) + pad_begin;
            memcpy(dst + i * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(eT));
        }

        for (int64_t i = 0; i < pad_end; i++) {
            int64_t copy_idx      = get_reflect_idx(input_length + i, input_length) + pad_begin;
            const int64_t dst_idx = pad_begin + input_length + i;
            memcpy(dst + dst_idx * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(eT));
        }
    } else if (_mode == PAD_MODE_EDGE) {
        for (int64_t i = 0; i < pad_begin; i++) {
            int64_t copy_idx = pad_begin;
            memcpy(dst + i * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(eT));
        }

        for (int64_t i = 0; i < pad_end; i++) {
            int64_t copy_idx      = pad_begin + input_length - 1;
            const int64_t dst_idx = pad_begin + input_length + i;
            memcpy(dst + dst_idx * stride_out, dst + copy_idx * stride_out, stride_out * sizeof(eT));
        }
    }
}

template <typename eT, int32_t c_blk, pad_mode_type_t _mode>
ppl::common::RetCode pad_nbcx_recursive_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    const int64_t *stride_in,
    const int64_t *stride_out,
    eT constant_value,
    const int64_t dim_idx,
    const int64_t c_dim_idx,
    const bool has_paralleled,
    eT *dst)
{
    const int64_t pad_begin = start_pads[dim_idx];
    const int64_t pad_end   = end_pads[dim_idx];

    const int64_t length    = dim_idx == c_dim_idx ? div_up(src_shape->GetDim(dim_idx), c_blk) : src_shape->GetDim(dim_idx);
    const int64_t dim_count = src_shape->GetDimCount();

    if (dim_idx == dim_count - 1) { // last dim
        pad_nbcx_last_dim_pad_begin_common<eT, c_blk, _mode>(src, pad_begin, length, constant_value, dst);
        memcpy(dst + pad_begin * c_blk, src, length * c_blk * sizeof(eT));
        pad_nbcx_last_dim_pad_end_common<eT, c_blk, _mode>(src, pad_begin, pad_end, length, constant_value, dst);
    } else {
        if (length > 1 && !has_paralleled) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                pad_nbcx_recursive_common<eT, c_blk, _mode>(
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
                pad_nbcx_recursive_common<eT, c_blk, _mode>(
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
        pad_nbcx_begin_end_common<eT, c_blk, _mode>(src, pad_begin, pad_end, length, stride_out[dim_idx], constant_value, dst);
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk, pad_mode_type_t _mode>
static ppl::common::RetCode pad_nbcx_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const int64_t *start_pads,
    const int64_t *end_pads,
    eT constant_value,
    eT *dst)
{
    const int64_t dim_count = dst_shape->GetDimCount();

    const int64_t c_dim_idx = 1;
    if (start_pads[c_dim_idx] != 0 || end_pads[c_dim_idx] != 0) { // pad on c will be changed to ndarray implement by opt_kernel
        return ppl::common::RC_UNSUPPORTED;
    }

    std::vector<int64_t> stride_in(dim_count, c_blk);
    std::vector<int64_t> stride_out(dim_count, c_blk);
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

    return pad_nbcx_recursive_common<eT, c_blk, _mode>(
        src_shape,
        dst_shape,
        src,
        start_pads,
        end_pads,
        stride_in.data(),
        stride_out.data(),
        constant_value,
        0,
        c_dim_idx,
        false,
        dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_PAD_NEON_PAD_NBCX_COMMON_H_