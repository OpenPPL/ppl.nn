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

#include <new>

#include "ppl/kernel/x86/fp32/fc.h"
#include "ppl/kernel/x86/fp32/fc/fma/fc_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

fc_fp32_algo_info fc_algo_selector::select_algo(const ppl::common::dataformat_t &src_format, const fc_fp32_param &param, const ppl::common::isa_t &isa_flags)
{
    (void)src_format;

    static fc_fp32_algo_info unknown_info = {
        fc_fp32_algo::UNKNOWN,
        ppl::common::ISA_UNKNOWN};

    if (isa_flags & ppl::common::ISA_X86_FMA) {
        return {
            fc_fp32_algo::STANDARD,
            ppl::common::ISA_X86_FMA};
    } else {
        return unknown_info;
    }
}

fc_fp32_manager *fc_algo_selector::gen_algo(const fc_fp32_param &param, const fc_fp32_algo_info &algo_info, ppl::common::Allocator *allocator)
{
    fc_fp32_manager *fc_mgr = nullptr;
    if (algo_info.algo_type == fc_fp32_algo::STANDARD &&
        algo_info.isa == ppl::common::ISA_X86_FMA) {
        fc_mgr = new fc_fp32_fma_manager(param, allocator);
    }

    return fc_mgr;
}

}}}; // namespace ppl::kernel::x86
