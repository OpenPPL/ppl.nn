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
#include "ppl/kernel/x86/fp32/transpose/avx/transpose_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reorder_n16cx_ndarray_fp32_avx(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst)
{
    if (src_shape->GetDataFormat() != ppl::common::DATAFORMAT_N16CX ||
        src_shape->GetDimCount() < 3) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t X        = src_shape->GetElementsExcludingPadding() / batch / channels;

    const int64_t simd_w   = 8;
    const int64_t c_blk    = 16;
    const int64_t padded_c = round_up(channels, c_blk);

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t b = 0; b < batch; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t c = 0; c < channels; c += c_blk) {
            for (int64_t x = 0; x < X; x += simd_w) {
                const int64_t c_eff = min<int64_t>(channels - c, c_blk);
                const int64_t x_eff = min<int64_t>(X - x, simd_w);
                for (int64_t mc = 0; mc < c_eff; mc += simd_w) {
                    const int64_t mc_eff = min<int64_t>(c_eff - mc, simd_w);
                    float *ldst          = dst + b * channels * X + (c + mc) * X + x;
                    const float *lsrc    = src + b * padded_c * X + c * X + x * c_blk + mc;
                    if (x_eff == simd_w && mc_eff == simd_w) {
                        transpose_8x8_fp32_avx(lsrc, c_blk, X, ldst);
                    } else {
                        for (int64_t xx = 0; xx < x_eff; ++xx) {
                            for (int64_t cc = 0; cc < mc_eff; ++cc) {
                                ldst[cc * X + xx] = lsrc[xx * c_blk + cc];
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
