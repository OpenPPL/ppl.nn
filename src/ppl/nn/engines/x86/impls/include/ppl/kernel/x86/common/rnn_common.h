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

#ifndef __ST_PPL_KERNEL_X86_FP32_RNN_COMMON_H_
#define __ST_PPL_KERNEL_X86_FP32_RNN_COMMON_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

typedef uint64_t rnn_direction_t;

class rnn_direction {
public:
    static const rnn_direction_t forward       = 0;
    static const rnn_direction_t reverse       = 1;
    static const rnn_direction_t bidirectional = 2;
};

class rnn_num_gate {
public:
    static const int64_t lstm = 4;
    static const int64_t gru  = 3;
};

}}}; // namespace ppl::kernel::x86

#endif
