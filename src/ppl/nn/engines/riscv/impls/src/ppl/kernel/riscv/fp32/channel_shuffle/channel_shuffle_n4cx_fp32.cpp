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

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/common/math.h"

namespace ppl { namespace kernel { namespace riscv {

inline void channel_shuffle_n4cx_pad_channel(
    const int64_t length,
    const int64_t c_eff,
    float* dst)
{
    const int64_t c_blk = 4;
    for (int64_t i = 0; i < length; i++) {
        for (int64_t c = c_eff; c < c_blk; c++) {
            dst[i * c_blk + c] = 0;
        }
    }
}

template <int32_t channels>
inline void channel_shuffle_n4cx_kernel(
    const float* src[4],
    const int64_t& length,
    float* dst)
{
    const int64_t c_blk = 4;
    for (int64_t l = 0; l < length; ++l) {
        if (channels > 0) dst[l * c_blk + 0] = src[0][l * c_blk];
        if (channels > 1) dst[l * c_blk + 1] = src[1][l * c_blk];
        if (channels > 2) dst[l * c_blk + 2] = src[2][l * c_blk];
        if (channels > 3) dst[l * c_blk + 3] = src[3][l * c_blk];
    }
}

ppl::common::RetCode channel_shuffle_n4cx_fp32(
    const ppl::common::TensorShape* src_shape,
    const float* src,
    const int32_t group,
    float* dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t height   = src_shape->GetDim(2);
    const int64_t width    = src_shape->GetDim(3);

    const int64_t c_blk              = 4;
    const int64_t channels_per_group = channels / group;
    const int64_t pad_c              = round_up(channels, c_blk);
    const int64_t _3D                = pad_c * height * width;
    const int64_t _2D                = height * width;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t oc = 0; oc < channels; oc += c_blk) {
            const int64_t oc_len_efficient = min(channels - oc, c_blk);
            float* base_dst                = dst + b * _3D * oc * _2D;
            const float* base_src[4]       = {0};
            for (int64_t i = 0; i < oc_len_efficient; i++) {
                const int64_t ic        = (oc + i) % group * channels_per_group + (oc + i) / group;
                const int64_t padded_ic = round(ic, c_blk);
                base_src[i]             = src + b * _3D + padded_ic * _2D + ic % c_blk;
            }
            if (oc_len_efficient == 4)
                channel_shuffle_n4cx_kernel<4>(base_src, _2D, base_dst);
            else if (oc_len_efficient == 3)
                channel_shuffle_n4cx_kernel<3>(base_src, _2D, base_dst);
            else if (oc_len_efficient == 2)
                channel_shuffle_n4cx_kernel<2>(base_src, _2D, base_dst);
            else if (oc_len_efficient == 1)
                channel_shuffle_n4cx_kernel<1>(base_src, _2D, base_dst);

            const bool pad_dst = channels != pad_c && oc + c_blk >= channels;
            if (pad_dst) {
                const int64_t round_dst_c = round(channels, c_blk);
                channel_shuffle_n4cx_pad_channel(_2D, channels - round_dst_c, dst + (b * pad_c + round_dst_c) * _2D);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

template <int32_t channels>
inline void channel_shuffle_n4cx_concat_split_kernel(
    const float* src[4],
    const int64_t& length,
    float* dst[4])
{
    const int64_t c_blk = 4;
    for (int64_t l = 0; l < length; l++) {
        if (channels > 0) dst[0][l * c_blk] = src[0][l * c_blk];
        if (channels > 1) dst[1][l * c_blk] = src[1][l * c_blk];
        if (channels > 2) dst[2][l * c_blk] = src[2][l * c_blk];
        if (channels > 3) dst[3][l * c_blk] = src[3][l * c_blk];
    }
}

ppl::common::RetCode channel_shuffle_n4cx_concat_split_fp32(
    const ppl::common::TensorShape* src0_shape,
    const ppl::common::TensorShape* src1_shape,
    const float* src0,
    const float* src1,
    const int32_t group,
    float* dst0,
    float* dst1_optional)
{
    const int64_t in_c1    = src0_shape->GetDim(1);
    const int64_t in_c2    = src1_shape->GetDim(1);
    const int64_t channels = in_c1 + in_c2;
    if (dst1_optional && channels % 2) {
        return ppl::common::RC_INVALID_VALUE;
    }
    float* dst1          = dst1_optional;
    const int64_t out_c1 = dst1 ? channels / 2 : channels;
    const int64_t out_c2 = dst1 ? channels / 2 : 0;

    const int64_t batch      = src0_shape->GetDim(0);
    const int64_t src_h      = src0_shape->GetDim(2);
    const int64_t src_w      = src0_shape->GetDim(3);
    const int64_t inner_dims = src_h * src_w;

    const int64_t c_blk              = 4;
    const int64_t channels_per_group = channels / group;
    const int64_t padded_in_c1       = round_up(in_c1, c_blk);
    const int64_t padded_in_c2       = round_up(in_c2, c_blk);
    const int64_t padded_out_c1      = round_up(out_c1, c_blk);
    const int64_t padded_out_c2      = round_up(out_c2, c_blk);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t oc = 0; oc < channels; oc += c_blk) {
            const int64_t oc_len_efficient = min(channels - oc, c_blk);
            const float* base_src[4]       = {0};
            float* base_dst[4]             = {0};
            for (int64_t i = 0; i < oc_len_efficient; i++) {
                const int64_t cur_oc = oc + i;
                const int64_t cur_ic = cur_oc % group * channels_per_group + cur_oc / group;
                const float* src_;
                float* dst_;
                if (cur_ic < in_c1) {
                    const int64_t round_ic = round(cur_ic, c_blk);
                    src_                   = src0 + b * padded_in_c1 * inner_dims + round_ic * inner_dims + (cur_ic - round_ic);
                } else {
                    const int64_t round_ic = round(cur_ic - in_c1, c_blk);
                    src_                   = src1 + b * padded_in_c2 * inner_dims + round_ic * inner_dims + (cur_ic - in_c1 - round_ic);
                }
                if (cur_oc < out_c1) {
                    const int64_t round_oc = round(cur_oc, c_blk);
                    dst_                   = dst0 + b * padded_out_c1 * inner_dims + round_oc * inner_dims + (cur_oc - round_oc);
                } else {
                    const int64_t round_oc = round(cur_oc - out_c1, c_blk);
                    dst_                   = dst1 + b * padded_out_c2 * inner_dims + round_oc * inner_dims + (cur_oc - out_c1 - round_oc);
                }
                base_src[i] = src_;
                base_dst[i] = dst_;
            }
            if (oc_len_efficient == 4)
                channel_shuffle_n4cx_concat_split_kernel<4>(base_src, inner_dims, base_dst);
            else if (oc_len_efficient == 3)
                channel_shuffle_n4cx_concat_split_kernel<3>(base_src, inner_dims, base_dst);
            else if (oc_len_efficient == 2)
                channel_shuffle_n4cx_concat_split_kernel<2>(base_src, inner_dims, base_dst);
            else if (oc_len_efficient == 1)
                channel_shuffle_n4cx_concat_split_kernel<1>(base_src, inner_dims, base_dst);

            const bool pad_dst0 = oc < out_c1 && oc + c_blk > out_c1;
            const bool pad_dst1 = oc + c_blk >= channels && padded_out_c2 != out_c2;
            if (pad_dst0) {
                const int64_t round_dst_c0 = round(out_c1, c_blk);
                channel_shuffle_n4cx_pad_channel(
                    inner_dims,
                    out_c1 - round_dst_c0,
                    dst0 + (b * padded_out_c1 + round_dst_c0) * inner_dims);
            }
            if (pad_dst1) {
                const int64_t round_dst_c1 = round(out_c2, c_blk);
                channel_shuffle_n4cx_pad_channel(
                    inner_dims,
                    out_c2 - round_dst_c1,
                    dst1 + (b * padded_out_c2 + round_dst_c1) * inner_dims);
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::riscv
