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

#ifndef __ST_PPL_KERNEL_X86_COMMON_RELATION_RELATION_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_RELATION_RELATION_COMMON_H_

namespace ppl { namespace kernel { namespace x86 {

enum relation_op_type_t {
    RELATION_GREATER          = 0,
    RELATION_GREATER_OR_EQUAL = 1,
    RELATION_LESS             = 2,
    RELATION_LESS_OR_EQUAL    = 3,
    RELATION_EQUAL            = 4,
    RELATION_NOT_EQUAL        = 5
};

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_RELATION_RELATION_COMMON_H_
