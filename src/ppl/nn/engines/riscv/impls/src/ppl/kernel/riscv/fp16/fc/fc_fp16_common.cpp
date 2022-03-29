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

#include "ppl/kernel/riscv/fp16/fc.h"
#include "ppl/kernel/riscv/fp16/fc/vec128/fc_fp16_vec128.h"
#include "ppl/kernel/riscv/fp16/fc/vec128/fc_ndarray_fp16_vec128.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

fc_common_algo_info fc_algo_selector_fp16::select_algo(const ppl::common::dataformat_t& src_format,
                                                       const fc_common_param& param)
{
    static fc_common_algo_info unknown_info = {fc_common_algo::unknown};

    if (false) {
    } else if (src_format == ppl::common::DATAFORMAT_NDARRAY) {
        return {
            fc_common_algo::standard,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATATYPE_FLOAT16,
            ppl::common::DATATYPE_FLOAT16};
    } else if (src_format == ppl::common::DATAFORMAT_N8CX) {
        return {
            fc_common_algo::standard,
            ppl::common::DATAFORMAT_N8CX,
            ppl::common::DATAFORMAT_N8CX,
            ppl::common::DATATYPE_FLOAT16,
            ppl::common::DATATYPE_FLOAT16};
    }

    return unknown_info;
}

fc_manager<__fp16>* fc_algo_selector_fp16::gen_algo(const fc_common_param& param, const fc_common_algo_info& algo_info, ppl::common::Allocator* allocator)
{
    fc_manager<__fp16>* fc_mgr = nullptr;
    if (algo_info.algo_type == fc_common_algo::standard && algo_info.input_format == ppl::common::DATAFORMAT_NDARRAY) {
        fc_mgr = new fc_ndarray_fp16_vec128_manager(param, allocator);
    } else if (algo_info.algo_type == fc_common_algo::standard && algo_info.input_format == ppl::common::DATAFORMAT_N8CX) {
        fc_mgr = new fc_fp16_vec128_manager(param, allocator);
    } else {
        LOG(ERROR) << "FC gen algo failed.";
    }

    return fc_mgr;
}

}}}; // namespace ppl::kernel::riscv
