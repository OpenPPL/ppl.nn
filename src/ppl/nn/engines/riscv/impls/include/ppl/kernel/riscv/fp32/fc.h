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

#ifndef __ST_PPL_KERNEL_RISCV_FP32_FC_H_
#define __ST_PPL_KERNEL_RISCV_FP32_FC_H_

#include <string>

#include "ppl/kernel/riscv/common/general_include.h"
#include "ppl/kernel/riscv/common/fc.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace riscv {

class fc_algo_selector_fp32 {
public:
    static fc_common_algo_info select_algo(const ppl::common::dataformat_t& src_format, const fc_common_param& param);
    static fc_manager<float>* gen_algo(const fc_common_param& param, const fc_common_algo_info& algo_info, ppl::common::Allocator* allocator);
};
}}}; // namespace ppl::kernel::riscv

#endif
