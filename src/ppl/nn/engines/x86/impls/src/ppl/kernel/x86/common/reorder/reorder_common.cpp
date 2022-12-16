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

bool reorder_ndarray_n16cx_may_inplace(const ppl::common::TensorShape *src_shape) {
    const int64_t c_blk  = 16;
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);

    if (channels % 16 != 0 || batch * channels <= 2 * PPL_OMP_MAX_THREADS() * c_blk) {
        return false;
    }

    const int64_t l3_cap = (ppl::common::GetCpuCacheL3() == 0) ? (2048 * 1024 * PPL_OMP_MAX_THREADS()) : ppl::common::GetCpuCacheL3();
    const int64_t l2_cap = (ppl::common::GetCpuCacheL2() == 0) ? (256 * 1024) : ppl::common::GetCpuCacheL2();

    const int64_t X_bytes  = src_shape->CalcBytesExcludingPadding() / batch / channels;
    const int64_t padded_c = round_up(channels, c_blk);

    const int64_t in_bytes = batch * X_bytes;

    if (in_bytes * (channels + padded_c) > l3_cap && c_blk * X_bytes < l2_cap * 0.8f) {
        return true;
    }

    return false;
}

bool reorder_n16cx_ndarray_may_inplace(const ppl::common::TensorShape *src_shape) {
    const int64_t c_blk  = 16;
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);

    if (batch * channels <= 2 * PPL_OMP_MAX_THREADS() * c_blk) {
        return false;
    }

    const int64_t l3_cap = (ppl::common::GetCpuCacheL3() == 0) ? (2048 * 1024 * PPL_OMP_MAX_THREADS()) : ppl::common::GetCpuCacheL3();
    const int64_t l2_cap = (ppl::common::GetCpuCacheL2() == 0) ? (256 * 1024) : ppl::common::GetCpuCacheL2();

    const int64_t X_bytes  = src_shape->CalcBytesExcludingPadding() / batch / channels;
    const int64_t padded_c = round_up(channels, c_blk);

    const int64_t in_bytes = batch * X_bytes;

    if (in_bytes * (channels + padded_c) > l3_cap && c_blk * X_bytes < l2_cap * 0.8f) {
        return true;
    }

    return false;
}

}}}; // namespace ppl::kernel::x86
