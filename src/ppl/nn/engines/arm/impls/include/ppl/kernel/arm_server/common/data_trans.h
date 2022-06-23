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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_COMMON_DATA_TRANS_H_
#define __ST_PPL_KERNEL_ARM_SERVER_COMMON_DATA_TRANS_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server {

ppl::common::RetCode N4cxToNdarrayFp32(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst);

ppl::common::RetCode NdarrayToN4cxFp32(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst);

#ifdef PPLNN_USE_ARMV8_2_FP16
ppl::common::RetCode N8cxToNdarrayFp16(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst);

ppl::common::RetCode NdarrayToN8cxFp16(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst);

ppl::common::RetCode NdarrayFp32ToN8cxFp16(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst);

ppl::common::RetCode N8cxFp16ToNdarrayFp32(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst);

ppl::common::RetCode Fp32ToFp16(const float* src, const int64_t len, __fp16* dst);

ppl::common::RetCode Fp16ToFp32(const __fp16* src, const int64_t len, float* dst);

ppl::common::RetCode N4cxFp32ToN8cxFp16(const float* src, int64_t batch, int64_t channels, int64_t height, int64_t width, __fp16* dst);

ppl::common::RetCode N8cxFp16ToN4cxFp32(const __fp16* src, int64_t batch, int64_t channels, int64_t height, int64_t width, float* dst);
#endif

}}} // namespace ppl::kernel::arm_server

#endif
