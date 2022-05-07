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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_CONV2D_H_
#define __ST_PPL_KERNEL_RISCV_FP16_CONV2D_H_

#include <string>
#include <float.h>

#include "ppl/nn/engines/riscv/engine_options.h"
#include "ppl/kernel/riscv/common/conv2d.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/common/allocator.h"
#include "ppl/common/sys.h"
#include "functional"

namespace ppl { namespace kernel { namespace riscv {

class conv2d_fp16_algo_selector : public conv2d_algo_selector<__fp16> {
public:
    static conv2d_common_algo_info select_best_algo(const void* filter, ppl::nn::TensorShape& src_shape, ppl::nn::TensorShape& dst_shape, const conv2d_common_param& param, ppl::common::Allocator* allocator, const ppl::nn::riscv::EngineOptions* engine_options);
    static conv2d_common_algo_info select_algo(const ppl::nn::TensorShape& input_shape,
                                               const conv2d_common_param& param,
                                               const ppl::nn::riscv::EngineOptions* engine_options);
    static conv2d_offline_manager<__fp16>* gen_algo(const conv2d_common_param& param,
                                                    const conv2d_common_algo_info& algo_info,
                                                    ppl::common::Allocator* allocator);
};

}}}; // namespace ppl::kernel::riscv

#endif // __ST_PPL_KERNEL_RISCV_FP16_CONV2D_H_
