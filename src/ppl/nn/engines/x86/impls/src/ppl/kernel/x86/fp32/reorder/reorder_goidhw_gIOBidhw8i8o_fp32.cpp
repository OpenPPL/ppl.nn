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

#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

#define CH_DT_BLK() 8

uint64_t reorder_goidhw_gIOBidhw8i8o_fp32_get_dst_size(
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t channels_blk)
{
    const int32_t ic_per_gp  = channels / group;
    const int32_t oc_per_gp  = num_output / group;
    const int32_t padded_ic  = round_up(ic_per_gp, CH_DT_BLK());
    const int32_t padded_oc  = round_up(oc_per_gp, CH_DT_BLK());
    const int32_t ic_big_blk = channels_blk;
    const int32_t ic_big_cnt = div_up(padded_ic, channels_blk);

    return uint64_t(group) * ic_big_cnt * padded_oc * kernel_d * kernel_h * kernel_w * ic_big_blk * sizeof(float);
}

ppl::common::RetCode reorder_goidhw_gIOBidhw8i8o_fp32(
    const float *src,
    const int32_t group,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_d,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t channels_blk,
    float *dst)
{
    const int32_t ic_per_gp  = channels / group;
    const int32_t oc_per_gp  = num_output / group;
    const int32_t padded_ic  = round_up(ic_per_gp, CH_DT_BLK());
    const int32_t padded_oc  = round_up(oc_per_gp, CH_DT_BLK());
    const int32_t ic_big_blk = channels_blk;
    const int32_t ic_big_cnt = div_up(padded_ic, channels_blk);

    for (int64_t g = 0; g < group; ++g) {
        for (int64_t ic_big = 0; ic_big < padded_ic; ic_big += ic_big_blk) {
            for (int64_t ocb = 0; ocb < padded_oc; ocb += CH_DT_BLK()) {
                const int64_t ic_big_eff = min<int64_t>(ic_per_gp - ic_big, ic_big_blk);
                const int64_t ocb_eff    = min<int64_t>(oc_per_gp - ocb, CH_DT_BLK());
                const float *base_src    = src + g * oc_per_gp * ic_per_gp * kernel_d * kernel_h * kernel_w + ocb * ic_per_gp * kernel_d * kernel_h * kernel_w + ic_big * kernel_d * kernel_h * kernel_w;
                float *base_dst          = dst + g * ic_big_cnt * padded_oc * ic_big_blk * kernel_d * kernel_h * kernel_w + ic_big * padded_oc * kernel_d * kernel_h * kernel_w + ocb * ic_big_blk * kernel_d * kernel_h * kernel_w;
                for (int64_t icb = 0; icb < ic_big_eff; icb += CH_DT_BLK()) {
                    const int64_t icb_eff = min<int64_t>(ic_big_eff - icb, CH_DT_BLK());
                    for (int64_t kd = 0; kd < kernel_d; ++kd) {
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                const float *l_src = base_src + icb * kernel_d * kernel_h * kernel_w + kd * kernel_h * kernel_w + kh * kernel_w + kw;
                                float *l_dst       = base_dst + icb * kernel_d * kernel_h * kernel_w * CH_DT_BLK() + kd * kernel_h * kernel_w * CH_DT_BLK() * CH_DT_BLK() + kh * kernel_w * CH_DT_BLK() * CH_DT_BLK() + kw * CH_DT_BLK() * CH_DT_BLK();
                                int64_t ic         = 0;
                                for (; ic < icb_eff; ++ic) {
                                    int64_t oc = 0;
                                    for (; oc < ocb_eff; ++oc) {
                                        l_dst[ic * CH_DT_BLK() + oc] = l_src[(oc * ic_per_gp + ic) * kernel_d * kernel_h * kernel_w];
                                    }
                                    for (; oc < CH_DT_BLK(); ++oc) {
                                        l_dst[ic * CH_DT_BLK() + oc] = 0;
                                    }
                                }
                                for (; ic < CH_DT_BLK(); ++ic) {
                                    memset(l_dst + ic * CH_DT_BLK(), 0, CH_DT_BLK() * sizeof(float));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
