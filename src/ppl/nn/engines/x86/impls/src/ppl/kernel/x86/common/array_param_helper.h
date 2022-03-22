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

#ifndef __ST_PPL_KERNEL_X86_COMMON_ARRAY_PARAM_HELPER_H_
#define __ST_PPL_KERNEL_X86_COMMON_ARRAY_PARAM_HELPER_H_

#include <stdint.h>

namespace ppl { namespace kernel { namespace x86 {

class array_param_helper
{
public:
    array_param_helper(int64_t *param) : param_(param) {}
    
    template<typename T> inline T& pick(const int64_t idx) {
        return *(T*)(param_ + idx);
    }

    template<typename T> inline T pick(const int64_t idx) const { 
        return *(T*)(param_ + idx);
    }

private:
    int64_t *param_;
};

}}}; // namespace ppl::kernel::x86

#endif
