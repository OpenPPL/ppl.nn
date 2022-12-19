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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_NBCX_COMMON_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, int32_t c_blk>
inline void channel_shuffle_nbcx_kernel_common(
    const eT *p_src[],
    const int64_t length,
    const int64_t channels,
    eT *p_dst[])
{
    for (int64_t i = 0; i < length; i++) {
        for (int64_t c = 0; c < channels; c++) {
            p_dst[c][i * c_blk] = p_src[c][i * c_blk];
        }
    }
}

template <typename eT, int32_t c_blk, int32_t channels>
inline void channel_shuffle_nbcx_kernel_template_common(
    const eT *p_src[],
    const int64_t length,
    eT *p_dst[])
{
    for (int64_t i = 0; i < length; i++) {
        if (channels > 0) p_dst[0][i * c_blk] = p_src[0][i * c_blk];
        if (channels > 1) p_dst[1][i * c_blk] = p_src[1][i * c_blk];
        if (channels > 2) p_dst[2][i * c_blk] = p_src[2][i * c_blk];
        if (channels > 3) p_dst[3][i * c_blk] = p_src[3][i * c_blk];
        if (channels > 4) p_dst[4][i * c_blk] = p_src[4][i * c_blk];
        if (channels > 5) p_dst[5][i * c_blk] = p_src[5][i * c_blk];
        if (channels > 6) p_dst[6][i * c_blk] = p_src[6][i * c_blk];
        if (channels > 7) p_dst[7][i * c_blk] = p_src[7][i * c_blk];
    }
}

template <typename eT, int32_t c_blk>
inline void channel_shuffle_nbcx_pad_channel(
    const int64_t length,
    const int64_t c_eff,
    eT *p_dst)
{
    for (int64_t i = 0; i < length; i++) {
        for (int64_t c = c_eff; c < c_blk; c++) {
            p_dst[i * c_blk + c] = 0;
        }
    }
}

template <typename eT, int32_t c_blk>
static ppl::common::RetCode channel_shuffle_nbcx_concat_split_common(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst0_shape,
    const ppl::common::TensorShape *dst1_shape,
    const eT *src0,
    const eT *src1,
    const int32_t group,
    eT *dst0,
    eT *dst1)
{
    const int64_t src_c0   = src0_shape->GetDim(1);
    const int64_t src_c1   = src1_shape->GetDim(1);
    const int64_t channels = src_c0 + src_c1;
    const int64_t dst_c0   = dst0_shape->GetDim(1);
    const int64_t dst_c1   = dst1_shape->GetDim(1);

    if (channels % group != 0 || dst_c0 + dst_c1 != channels) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch = src0_shape->GetDim(0);
    int64_t inner_dims  = 1;
    for (int64_t i = 2; i < src0_shape->GetDimCount(); i++) {
        inner_dims *= src0_shape->GetDim(i);
    }

    const int64_t channels_per_group = channels / group;

    const int64_t padded_src_c0 = round_up(src_c0, c_blk);
    const int64_t padded_src_c1 = round_up(src_c1, c_blk);
    const int64_t padded_dst_c0 = round_up(dst_c0, c_blk);
    const int64_t padded_dst_c1 = round_up(dst_c1, c_blk);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t n = 0; n < batch; n++) {
        for (int64_t ocb = 0; ocb < channels; ocb += c_blk) {
            const int64_t ocb_len_eff = min(channels - ocb, (int64_t)c_blk);
            const eT *p_src[c_blk]    = {0};
            eT *p_dst[c_blk]          = {0};
            for (int64_t i = 0; i < ocb_len_eff; i++) {
                const int64_t cur_oc = ocb + i;
                const int64_t cur_ic = cur_oc % group * channels_per_group + cur_oc / group;
                if (cur_ic < src_c0) {
                    const int64_t round_ic = round(cur_ic, c_blk);
                    p_src[i]               = src0 + (n * padded_src_c0 + round_ic) * inner_dims + cur_ic - round_ic;
                } else {
                    const int64_t round_ic = round(cur_ic - src_c0, c_blk);
                    p_src[i]               = src1 + (n * padded_src_c1 + round_ic) * inner_dims + cur_ic - src_c0 - round_ic;
                }
                if (cur_oc < dst_c0) {
                    const int64_t round_oc = round(cur_oc, c_blk);
                    p_dst[i]               = dst0 + (n * padded_dst_c0 + round_oc) * inner_dims + cur_oc - round_oc;
                } else {
                    const int64_t round_oc = round(cur_oc - dst_c0, c_blk);
                    p_dst[i]               = dst1 + (n * padded_dst_c1 + round_oc) * inner_dims + cur_oc - dst_c0 - round_oc;
                }
            }

            switch (ocb_len_eff) {
                case 0: break;
                case 1: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 1>(p_src, inner_dims, p_dst); break;
                case 2: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 2>(p_src, inner_dims, p_dst); break;
                case 3: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 3>(p_src, inner_dims, p_dst); break;
                case 4: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 4>(p_src, inner_dims, p_dst); break;
                case 5: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 5>(p_src, inner_dims, p_dst); break;
                case 6: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 6>(p_src, inner_dims, p_dst); break;
                case 7: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 7>(p_src, inner_dims, p_dst); break;
                case 8: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 8>(p_src, inner_dims, p_dst); break;
                default: channel_shuffle_nbcx_kernel_common<eT, c_blk>(p_src, inner_dims, ocb_len_eff, p_dst); break;
            }

            const bool pad_dst0 = ocb < dst_c0 && ocb + c_blk > dst_c0;
            const bool pad_dst1 = ocb + c_blk >= channels && padded_dst_c1 != dst_c1;
            if (pad_dst0) {
                const int64_t round_dst_c0 = round(dst_c0, c_blk);
                channel_shuffle_nbcx_pad_channel<eT, c_blk>(inner_dims, dst_c0 - round_dst_c0, dst0 + (n * padded_dst_c0 + round_dst_c0) * inner_dims);
            }
            if (pad_dst1) {
                const int64_t round_dst_c1 = round(dst_c1, c_blk);
                channel_shuffle_nbcx_pad_channel<eT, c_blk>(inner_dims, dst_c1 - round_dst_c1, dst1 + (n * padded_dst_c1 + round_dst_c1) * inner_dims);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk>
static ppl::common::RetCode channel_shuffle_nbcx_concat_common(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    const int32_t group,
    eT *dst)
{
    const int64_t src_c0   = src0_shape->GetDim(1);
    const int64_t src_c1   = src1_shape->GetDim(1);
    const int64_t channels = src_c0 + src_c1;
    const int64_t dst_c    = dst_shape->GetDim(1);

    if (channels % group != 0 || dst_c != channels) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch = src0_shape->GetDim(0);
    int64_t inner_dims  = 1;
    for (int64_t i = 2; i < src0_shape->GetDimCount(); i++) {
        inner_dims *= src0_shape->GetDim(i);
    }

    const int64_t channels_per_group = channels / group;

    const int64_t padded_src_c0 = round_up(src_c0, c_blk);
    const int64_t padded_src_c1 = round_up(src_c1, c_blk);
    const int64_t padded_dst_c  = round_up(dst_c, c_blk);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t n = 0; n < batch; n++) {
        for (int64_t ocb = 0; ocb < channels; ocb += c_blk) {
            const int64_t ocb_len_eff = min(channels - ocb, (int64_t)c_blk);
            const eT *p_src[c_blk]    = {0};
            eT *p_dst[c_blk]          = {0};
            for (int64_t i = 0; i < ocb_len_eff; i++) {
                const int64_t cur_oc = ocb + i;
                const int64_t cur_ic = cur_oc % group * channels_per_group + cur_oc / group;
                if (cur_ic < src_c0) {
                    const int64_t round_ic = round(cur_ic, c_blk);
                    p_src[i]               = src0 + (n * padded_src_c0 + round_ic) * inner_dims + cur_ic - round_ic;
                } else {
                    const int64_t round_ic = round(cur_ic - src_c0, c_blk);
                    p_src[i]               = src1 + (n * padded_src_c1 + round_ic) * inner_dims + cur_ic - src_c0 - round_ic;
                }
                const int64_t round_oc = round(cur_oc, c_blk);
                p_dst[i]               = dst + (n * padded_dst_c + round_oc) * inner_dims + cur_oc - round_oc;
            }

            switch (ocb_len_eff) {
                case 0: break;
                case 1: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 1>(p_src, inner_dims, p_dst); break;
                case 2: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 2>(p_src, inner_dims, p_dst); break;
                case 3: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 3>(p_src, inner_dims, p_dst); break;
                case 4: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 4>(p_src, inner_dims, p_dst); break;
                case 5: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 5>(p_src, inner_dims, p_dst); break;
                case 6: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 6>(p_src, inner_dims, p_dst); break;
                case 7: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 7>(p_src, inner_dims, p_dst); break;
                case 8: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 8>(p_src, inner_dims, p_dst); break;
                default: channel_shuffle_nbcx_kernel_common<eT, c_blk>(p_src, inner_dims, ocb_len_eff, p_dst); break;
            }

            const bool pad_dst = ocb + c_blk >= channels && padded_dst_c != dst_c;
            if (pad_dst) {
                const int64_t round_dst_c = round(dst_c, c_blk);
                channel_shuffle_nbcx_pad_channel<eT, c_blk>(inner_dims, dst_c - round_dst_c, dst + (n * padded_dst_c + round_dst_c) * inner_dims);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk>
static ppl::common::RetCode channel_shuffle_nbcx_common(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src,
    const int32_t group,
    eT *dst)
{
    const int64_t channels = src_shape->GetDim(1);

    if (channels % group != 0 || channels != dst_shape->GetDim(1)) {
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t batch = src_shape->GetDim(0);
    int64_t inner_dims  = 1;
    for (int64_t i = 2; i < src_shape->GetDimCount(); i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    const int64_t channels_per_group = channels / group;

    const int64_t padded_channels = round_up(channels, c_blk);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t n = 0; n < batch; n++) {
        for (int64_t ocb = 0; ocb < channels; ocb += c_blk) {
            const int64_t ocb_len_eff = min(channels - ocb, (int64_t)c_blk);
            const eT *p_src[c_blk]    = {0};
            eT *p_dst[c_blk]          = {0};
            for (int64_t i = 0; i < ocb_len_eff; i++) {
                const int64_t cur_oc   = ocb + i;
                const int64_t cur_ic   = cur_oc % group * channels_per_group + cur_oc / group;
                const int64_t round_ic = round(cur_ic, c_blk);
                const int64_t round_oc = round(cur_oc, c_blk);
                p_src[i]               = src + (n * padded_channels + round_ic) * inner_dims + cur_ic - round_ic;
                p_dst[i]               = dst + (n * padded_channels + round_oc) * inner_dims + cur_oc - round_oc;
            }

            switch (ocb_len_eff) {
                case 0: break;
                case 1: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 1>(p_src, inner_dims, p_dst); break;
                case 2: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 2>(p_src, inner_dims, p_dst); break;
                case 3: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 3>(p_src, inner_dims, p_dst); break;
                case 4: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 4>(p_src, inner_dims, p_dst); break;
                case 5: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 5>(p_src, inner_dims, p_dst); break;
                case 6: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 6>(p_src, inner_dims, p_dst); break;
                case 7: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 7>(p_src, inner_dims, p_dst); break;
                case 8: channel_shuffle_nbcx_kernel_template_common<eT, c_blk, 8>(p_src, inner_dims, p_dst); break;
                default: channel_shuffle_nbcx_kernel_common<eT, c_blk>(p_src, inner_dims, ocb_len_eff, p_dst); break;
            }

            const bool pad_dst = ocb + c_blk >= channels && padded_channels != channels;
            if (pad_dst) {
                const int64_t round_c = round(channels, c_blk);
                channel_shuffle_nbcx_pad_channel<eT, c_blk>(inner_dims, channels - round_c, dst + (n * padded_channels + round_c) * inner_dims);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}} // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_NBCX_COMMON_H_