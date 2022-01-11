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

#ifndef PPL_ARM_SERVER_KERNEL_INCLUDE_DETECT_CPU_AARCH64_H_
#define PPL_ARM_SERVER_KERNEL_INCLUDE_DETECT_CPU_AARCH64_H_

#include <stdint.h>
bool ppl_arm_server_check_taishan_v110();
bool ppl_arm_server_check_neoverse_n1();
bool ppl_arm_server_check_phytium_();

bool ppl_arm_server_check_ext_asimd();
bool ppl_arm_server_check_ext_i8mm();
bool ppl_arm_server_check_fp16();
bool ppl_arm_server_check_bf16();

#endif
