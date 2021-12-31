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

#include "ppl/common/log.h"

void CvtFp32ToFp16(int64_t counts, void const* src, void* dst) {
    LOG(DEBUG) << "fp32 to fp16";
    auto src_ptr = (float*)src;
    auto dst_ptr = (__fp16*)dst;
    for (int64_t i = 0; i < counts; i += 1) {
        dst_ptr[i] = src_ptr[i];
    }
}

void CvtFp16ToFp32(int64_t counts, void const* src, void* dst) {
    LOG(DEBUG) << "fp16 to fp32";
    auto src_ptr = (__fp16*)src;
    auto dst_ptr = (float*)dst;
    for (int64_t i = 0; i < counts; i += 1) {
        dst_ptr[i] = src_ptr[i];
    }
}
