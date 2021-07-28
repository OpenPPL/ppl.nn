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

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode reorder_n8cx_ndarray_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    float *dst)
{
    if (src_shape->GetDataFormat() != ppl::common::DATAFORMAT_N8CX ||
        src_shape->GetDimCount() < 3) {
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t X        = src_shape->GetElementsExcludingPadding() / batch / channels;

    const int64_t c_blk    = 8;
    const int64_t padded_c = round_up(channels, c_blk);

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t b = 0; b < batch; ++b) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t c = 0; c < channels; c += c_blk) {
            const int64_t c_eff = min<int64_t>(channels - c, c_blk);
            float *ldst         = dst + b * channels * X + c * X;
            const float *lsrc   = src + b * padded_c * X + c * X;
            for (int64_t x = 0; x < X; ++x) {
                for (int64_t cc = 0; cc < c_eff; ++cc) {
                    ldst[cc * X + x] = lsrc[x * c_blk + cc];
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
