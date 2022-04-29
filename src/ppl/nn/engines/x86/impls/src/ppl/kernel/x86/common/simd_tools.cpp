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

namespace ppl { namespace kernel { namespace x86 {

void set_denormals_zero(const int32_t on) {
    if (ppl::common::GetCpuISA() & ppl::common::ISA_X86_SSE) {
        PRAGMA_OMP_PARALLEL()
        {
            if (on) {
                _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
                _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
            } else {
                _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
                _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
            }
        }
    }
}

}}};
