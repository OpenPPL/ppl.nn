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

#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_COMMON_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_COMMON_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

typedef int32_t gemm_v_type_t;
class gemm_v_type {
public:
    static const gemm_v_type_t EMPTY = 0;
    static const gemm_v_type_t SCALAR = 1;
    static const gemm_v_type_t COL_VEC = 2;
    static const gemm_v_type_t ROW_VEC = 3;
};

typedef int32_t gemm_m_type_t;
class gemm_m_type {
public:
    static const gemm_m_type_t EMPTY = 0;
    static const gemm_m_type_t NOTRANS = 1;
    static const gemm_m_type_t TRANS = 2;
    static const gemm_m_type_t PACKED = 3;
};

typedef int32_t gemm_post_t;
class gemm_post {
public:
    static const gemm_post_t NONE = 0;
    static const gemm_post_t RELU = 1;
    static const gemm_post_t RELU6 = 2;
};

}}}; // namespace ppl::kernel::x86

#endif
