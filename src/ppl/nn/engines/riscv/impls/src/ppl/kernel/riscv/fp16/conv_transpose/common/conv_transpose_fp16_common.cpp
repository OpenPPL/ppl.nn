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
#include <chrono>
#include "ppl/kernel/riscv/fp16/conv_transpose.h"
#include "ppl/common/log.h"
#include "ppl/common/types.h"

using namespace ppl::common;

namespace ppl { namespace kernel { namespace riscv {

conv_transpose_common_algo_info conv_transpose_fp16_algo_selector::select_algo(const ppl::nn::riscv::EngineOptions* engine_options)
{
    return {DATAFORMAT_N8CX, DATAFORMAT_N8CX, DATATYPE_FLOAT16, DATATYPE_FLOAT16};
}

}}}; // namespace ppl::kernel::riscv
