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

#ifndef __ST_HPC_PPL_ARM_FP16FP32_H_
#define __ST_HPC_PPL_ARM_FP16FP32_H_
#ifdef __cplusplus
extern "C" {
#endif //! cplusplus

void CvtFp32ToFp16(int counts, void const *src, void *dst);
void CvtFp16ToFp32(int counts, void const *src, void *dst);

#ifdef __cplusplus
}
#endif //! cplusplus
#endif //! __ST_HPC_PPL_ARM_FP16FP32_H_
