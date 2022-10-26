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

#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_COMMON_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_COMMON_H_

#include "ppl/kernel/x86/common/general_include.h"
#include "ppl/kernel/x86/common/conv_common.h"

namespace ppl { namespace kernel { namespace x86 {
struct conv2d_param {
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t dilation_h;
    int64_t dilation_w;
    int64_t pad_h;
    int64_t pad_w;
    int64_t channels;
    int64_t num_output;
    int64_t group;
    conv_fuse_flag_t fuse_flag;

    float sparse_level() const
    {
        // TODO: are there any better index for sparse_level?
        const int32_t sparse_h = stride_h * dilation_h;
        const int32_t sparse_w = stride_w * dilation_w;
        return float(sparse_h * sparse_w) / float(kernel_h * kernel_w);
    }

    bool is_depthwise() const
    {
        return true &&
               group != 1 &&
               group == channels &&
               group == num_output;
    }

    bool is_pointwise() const
    {
        return true &&
               kernel_h == 1 &&
               kernel_w == 1 &&
               pad_h == 0 &&
               pad_w == 0 &&
               dilation_h == 1 &&
               dilation_w == 1 &&
               !is_depthwise();
    }
};

typedef uint32_t conv2d_algo_t;

class conv2d_algo {
public:
    static const conv2d_algo_t UNKNOWN         = 0;
    static const conv2d_algo_t IMPLICIT_GEMM   = 1;
    static const conv2d_algo_t GEMM_DIRECT     = 2;
    static const conv2d_algo_t DEPTHWISE       = 3;
    static const conv2d_algo_t IM2COL_GEMM     = 4;
    static const conv2d_algo_t DIRECT          = 5;
    static const conv2d_algo_t WINOGRAD_B2F3   = 32;
    static const conv2d_algo_t WINOGRAD_B4F3   = 33;
    static const conv2d_algo_t WINOGRAD_B6F3   = 34;
    static const conv2d_algo_t WINOGRAD_B2F5S2 = 35;
};

struct conv2d_algo_info {
    conv2d_algo_t algo_type;
    ppl::common::isa_t isa;
    ppl::common::dataformat_t input_format;
    ppl::common::dataformat_t output_format;
};

}}}; // namespace ppl::kernel::x86

#endif
