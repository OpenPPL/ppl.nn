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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_SPECIALIZATIONS_H_
#define __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_SPECIALIZATIONS_H_

#include <arm_neon.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/pad_channel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, int32_t c_blk>
inline void channel_shuffle_nbcx_pad_channel_c_blk_half(
    const int64_t length,
    eT *p_dst)
{
    constexpr int32_t c_blk_half = c_blk / 2; // use half of c_blk to match more case
    constexpr int32_t eN         = c_blk_half;
    typedef typename DT<eT, eN>::vecDT vecType;

    const vecType v_zero = vdup_n<eT, eN>(0);

    for (int64_t i = 0; i < length; i++) {
        vst<eT, eN>(p_dst + i * c_blk + c_blk_half, v_zero);
    }
}

/** specialization of channel shuffle with following conditions:
 * 1. NBCX data format
 * 2. fuse concat & split
 * 3. concat & split has two input/output with same channels
 * 4. group == 2
 * 5. channels_per_group % (c_blk / 2) == 0
 * 6. c_blk_half % group == 0
 * this specialization will be used by shufflenetv2
 * for fp32 n4cx, channels_per_group = 58, 116, 232 will use this
 * for fp16 n8cx, channels_per_group = 116, 232 will use this
 */
template <typename eT, int32_t c_blk>
static ppl::common::RetCode channel_shuffle_nbcx_concat_split_same2group_common(
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
    constexpr int32_t c_blk_half = c_blk / 2; // use half of c_blk to match more case
    constexpr int32_t eN         = c_blk_half;
    typedef typename DT<eT, eN>::vecDT vecType;

    if (group != 2) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t src_c0   = src0_shape->GetDim(1);
    const int64_t src_c1   = src1_shape->GetDim(1);
    const int64_t channels = src_c0 + src_c1;
    const int64_t dst_c0   = dst0_shape->GetDim(1);
    const int64_t dst_c1   = dst1_shape->GetDim(1);

    if (channels % group != 0 || dst_c0 + dst_c1 != channels) {
        return ppl::common::RC_INVALID_VALUE;
    }
    if (src_c0 != src_c1 || dst_c0 != dst_c1) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (channels % group != 0) {
        return ppl::common::RC_INVALID_VALUE;
    }
    const int64_t channels_per_group = channels / group;
    if (channels_per_group % c_blk_half != 0) {
        return ppl::common::RC_UNSUPPORTED;
    }
    if (c_blk_half % group != 0) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t batch = src0_shape->GetDim(0);
    int64_t inner_dims  = 1;
    for (int64_t i = 2; i < src0_shape->GetDimCount(); i++) {
        inner_dims *= src0_shape->GetDim(i);
    }

    const int64_t padded_channels_per_group = round_up(channels_per_group, c_blk);

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for (int64_t n = 0; n < batch; n++) {
        for (int64_t c = 0; c < padded_channels_per_group; c += c_blk) {
            const int64_t c_eff = min(channels_per_group - c, (int64_t)c_blk);

            const eT *p_src0  = src0 + (n * padded_channels_per_group + c) * inner_dims;
            const eT *p_src1  = src1 + (n * padded_channels_per_group + c) * inner_dims;
            const int64_t oc0 = c * group + c_blk_half / group * group * 0;
            const int64_t oc1 = c * group + c_blk_half / group * group * 1;
            const int64_t oc2 = c * group + c_blk_half / group * group * 2;
            const int64_t oc3 = c * group + c_blk_half / group * group * 3;
            eT *p_dst0        = oc0 < channels_per_group ? dst0 + (n * padded_channels_per_group + round(oc0, c_blk)) * inner_dims + oc0 % c_blk : dst1 + (n * padded_channels_per_group + round(oc0 - channels_per_group, c_blk)) * inner_dims + (oc0 - channels_per_group) % c_blk;
            eT *p_dst1        = oc1 < channels_per_group ? dst0 + (n * padded_channels_per_group + round(oc1, c_blk)) * inner_dims + oc1 % c_blk : dst1 + (n * padded_channels_per_group + round(oc1 - channels_per_group, c_blk)) * inner_dims + (oc1 - channels_per_group) % c_blk;
            eT *p_dst2        = oc2 < channels_per_group ? dst0 + (n * padded_channels_per_group + round(oc2, c_blk)) * inner_dims + oc2 % c_blk : dst1 + (n * padded_channels_per_group + round(oc2 - channels_per_group, c_blk)) * inner_dims + (oc2 - channels_per_group) % c_blk;
            eT *p_dst3        = oc3 < channels_per_group ? dst0 + (n * padded_channels_per_group + round(oc3, c_blk)) * inner_dims + oc3 % c_blk : dst1 + (n * padded_channels_per_group + round(oc3 - channels_per_group, c_blk)) * inner_dims + (oc3 - channels_per_group) % c_blk;

            if (c_eff == c_blk) {
                for (int64_t i = 0; i < inner_dims; i++) {
                    vecType v_src_0 = vld<eT, eN>(p_src0 + i * c_blk + c_blk_half * 0);
                    vecType v_src_1 = vld<eT, eN>(p_src0 + i * c_blk + c_blk_half * 1);
                    vecType v_src_2 = vld<eT, eN>(p_src1 + i * c_blk + c_blk_half * 0);
                    vecType v_src_3 = vld<eT, eN>(p_src1 + i * c_blk + c_blk_half * 1);

                    vecType v_dst_0 = vzip1<vecType>(v_src_0, v_src_2);
                    vecType v_dst_1 = vzip2<vecType>(v_src_0, v_src_2);
                    vecType v_dst_2 = vzip1<vecType>(v_src_1, v_src_3);
                    vecType v_dst_3 = vzip2<vecType>(v_src_1, v_src_3);

                    vst<eT, eN>(p_dst0 + i * c_blk, v_dst_0);
                    vst<eT, eN>(p_dst1 + i * c_blk, v_dst_1);
                    vst<eT, eN>(p_dst2 + i * c_blk, v_dst_2);
                    vst<eT, eN>(p_dst3 + i * c_blk, v_dst_3);
                }
            } else if (c_eff == c_blk_half) {
                for (int64_t i = 0; i < inner_dims; i++) {
                    vecType v_src_0 = vld<eT, eN>(p_src0 + i * c_blk + c_blk_half * 0);
                    vecType v_src_1 = vld<eT, eN>(p_src1 + i * c_blk + c_blk_half * 0);

                    vecType v_dst_0 = vzip1<vecType>(v_src_0, v_src_1);
                    vecType v_dst_1 = vzip2<vecType>(v_src_0, v_src_1);

                    vst<eT, eN>(p_dst0 + i * c_blk, v_dst_0);
                    vst<eT, eN>(p_dst1 + i * c_blk, v_dst_1);
                }
            }

            const int64_t start_oc = c * group;
            const int64_t end_oc   = (c + c_blk) * group;
            const bool pad_dst0    = start_oc < dst_c0 && end_oc > dst_c0 && padded_channels_per_group != dst_c0;
            const bool pad_dst1    = c + c_blk >= padded_channels_per_group && padded_channels_per_group != dst_c1;
            if (pad_dst0) {
                const int64_t round_dst_c0 = round(dst_c0, c_blk);
                channel_shuffle_nbcx_pad_channel_c_blk_half<eT, c_blk>(inner_dims, dst0 + (n * padded_channels_per_group + round_dst_c0) * inner_dims);
            }
            if (pad_dst1) {
                const int64_t round_dst_c1 = round(dst_c1, c_blk);
                channel_shuffle_nbcx_pad_channel_c_blk_half<eT, c_blk>(inner_dims, dst1 + (n * padded_channels_per_group + round_dst_c1) * inner_dims);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}} // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_CHANNEL_SHUFFLE_NEON_CHANNEL_SHUFFLE_SPECIALIZATIONS_H_