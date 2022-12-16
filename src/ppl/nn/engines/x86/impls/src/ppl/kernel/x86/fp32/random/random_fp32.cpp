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

#include <random>
#include <chrono>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode random_uniform_fp32(
    const ppl::common::TensorShape *output_shape,
    const float *seed,
    const float high,
    const float low,
    float *output)
{
    const int64_t n_elem    = output_shape->CalcElementsIncludingPadding();
    const uint32_t time_val = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    PRAGMA_OMP_PARALLEL()
    {
        auto local_seed = seed ? static_cast<uint32_t>(*seed) : (time_val * (PPL_OMP_THREAD_ID() + 1));
        std::default_random_engine gen(local_seed);
        std::uniform_real_distribution<float> dis(low, high);
        PRAGMA_OMP_FOR()
        for (int64_t n = 0; n < n_elem; ++n) {
            output[n] = dis(gen);
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
