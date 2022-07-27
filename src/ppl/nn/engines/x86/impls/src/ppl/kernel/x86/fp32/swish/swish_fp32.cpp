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
#include "ppl/kernel/x86/fp32/swish.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode swish_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *x_shape,
    const float *x,
    const float beta,
    float *y)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return swish_fp32_avx512(x_shape, x, beta, y);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return swish_fp32_fma(x_shape, x, beta, y);
    }
    return swish_fp32_sse(x_shape, x, beta, y);

}

}}}; // namespace ppl::kernel::x86
