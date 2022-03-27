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

#ifndef __ST_PPL_KERNEL_X86_COMMON_AVERAGEPOOL2D_AVERAGEPOOL2D_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_AVERAGEPOOL2D_AVERAGEPOOL2D_COMMON_H_

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

struct averagepool2d_param {
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t pad_h;
    int64_t pad_w;

    int64_t batch;
    int64_t channels;
    int64_t src_h;
    int64_t src_w;
    int64_t dst_h;
    int64_t dst_w;
};

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_AVERAGEPOOL2D_AVERAGEPOOL2D_COMMON_H_
