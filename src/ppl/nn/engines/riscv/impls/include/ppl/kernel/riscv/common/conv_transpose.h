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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_CONV_TRANSPOSE_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_CONV_TRANSPOSE_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

struct conv_transpose_common_algo_info {
    ppl::common::dataformat_t input_format;
    ppl::common::dataformat_t output_format;
    ppl::common::datatype_t input_data_type;
    ppl::common::datatype_t output_data_type;
};

}}}; // namespace ppl::kernel::riscv

#endif
