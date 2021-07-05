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

#include <immintrin.h>
#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/math.h"

namespace ppl { namespace kernel { namespace x86 {

template<int32_t channels>
inline void channel_shuffle_n16cx_kernel(
    const float *src[16],
    const int64_t &length,
    float *dst)
{
    const int64_t c_blk = 16;
    for (int64_t l = 0; l < length; ++l) {
        if (channels > 0 ) dst[l * c_blk + 0 ] = src[0 ][l * c_blk];
        if (channels > 1 ) dst[l * c_blk + 1 ] = src[1 ][l * c_blk];
        if (channels > 2 ) dst[l * c_blk + 2 ] = src[2 ][l * c_blk];
        if (channels > 3 ) dst[l * c_blk + 3 ] = src[3 ][l * c_blk];

        if (channels > 4 ) dst[l * c_blk + 4 ] = src[4 ][l * c_blk];
        if (channels > 5 ) dst[l * c_blk + 5 ] = src[5 ][l * c_blk];
        if (channels > 6 ) dst[l * c_blk + 6 ] = src[6 ][l * c_blk];
        if (channels > 7 ) dst[l * c_blk + 7 ] = src[7 ][l * c_blk];

        if (channels > 8 ) dst[l * c_blk + 8 ] = src[8 ][l * c_blk];
        if (channels > 9 ) dst[l * c_blk + 9 ] = src[9 ][l * c_blk];
        if (channels > 10) dst[l * c_blk + 10] = src[10][l * c_blk];
        if (channels > 11) dst[l * c_blk + 11] = src[11][l * c_blk];

        if (channels > 12) dst[l * c_blk + 12] = src[12][l * c_blk];
        if (channels > 13) dst[l * c_blk + 13] = src[13][l * c_blk];
        if (channels > 14) dst[l * c_blk + 14] = src[14][l * c_blk];
        if (channels > 15) dst[l * c_blk + 15] = src[15][l * c_blk];
    }
};

ppl::common::RetCode channel_shuffle_n16cx_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const int32_t group,
    float *dst)
{
    const int64_t batch = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t height = src_shape->GetDim(2);
    const int64_t width = src_shape->GetDim(3);

    const int64_t c_blk              = 16;
    const int64_t channels_per_group = channels / group;
    const int64_t _3D                = round_up(channels, c_blk) * height * width;
    const int64_t _2D                = height * width;

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t oc = 0; oc < channels; oc += c_blk) {
            const int64_t oc_len_eff = min(channels - oc, c_blk);
            float *base_dst          = dst + b * _3D + oc * _2D;
            const float *base_src[16] = {0};
            for (int64_t i = 0; i < oc_len_eff; ++i) {
                const int64_t ic        = (oc + i) % group * channels_per_group + (oc + i) / group;
                const int64_t padded_ic = round(ic, c_blk);
                base_src[i]             = src + b * _3D + padded_ic * _2D + ic % c_blk;
            }
            if      (oc_len_eff == 16) channel_shuffle_n16cx_kernel<16>(base_src, _2D, base_dst);
            else if (oc_len_eff == 15) channel_shuffle_n16cx_kernel<15>(base_src, _2D, base_dst);
            else if (oc_len_eff == 14) channel_shuffle_n16cx_kernel<14>(base_src, _2D, base_dst);
            else if (oc_len_eff == 15) channel_shuffle_n16cx_kernel<13>(base_src, _2D, base_dst);
            else if (oc_len_eff == 12) channel_shuffle_n16cx_kernel<12>(base_src, _2D, base_dst);
            else if (oc_len_eff == 11) channel_shuffle_n16cx_kernel<11>(base_src, _2D, base_dst);
            else if (oc_len_eff == 10) channel_shuffle_n16cx_kernel<10>(base_src, _2D, base_dst);
            else if (oc_len_eff == 9 ) channel_shuffle_n16cx_kernel<9 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 8 ) channel_shuffle_n16cx_kernel<8 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 7 ) channel_shuffle_n16cx_kernel<7 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 6 ) channel_shuffle_n16cx_kernel<6 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 5 ) channel_shuffle_n16cx_kernel<5 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 4 ) channel_shuffle_n16cx_kernel<4 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 3 ) channel_shuffle_n16cx_kernel<3 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 2 ) channel_shuffle_n16cx_kernel<2 >(base_src, _2D, base_dst);
            else if (oc_len_eff == 1 ) channel_shuffle_n16cx_kernel<1 >(base_src, _2D, base_dst);
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::x86
