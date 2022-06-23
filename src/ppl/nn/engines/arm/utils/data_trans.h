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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_UTILS_DATA_TRANS_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_UTILS_DATA_TRANS_H_

#ifdef PPLNN_USE_ARMV8_2_FP16

#include "ppl/nn/engines/arm/utils/fp16fp32_cvt.h"

void Fp32ToFp16(const float* src, int len, __fp16* dst);

void Fp16ToFp32(const __fp16* src, int len, float* dst);

#endif

#endif
