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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_ARM_ENGINE_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_ARM_ENGINE_OPTIONS_H_

#include "ppl/nn/common/common.h"
#include "ppl/common/types.h"
#include "ppl/nn/engines/arm/arm_options.h"
#include <stdint.h>

namespace ppl { namespace nn {

struct PPLNN_PUBLIC ArmEngineOptions final {
    uint32_t mm_policy = ARM_MM_COMPACT;
    uint32_t forward_precision = ppl::common::DATATYPE_FLOAT32;
    uint32_t graph_optimization_level = ARM_OPT_ENABLE_ALL;
    uint32_t winograd_level = ARM_WG_ON;
    uint32_t dynamic_tuning_level = ARM_TUNING_SELECT_ALGO;
    int32_t  numa_node_id = -1; // bind engine to speicified numa node, range [0, numa_max_node). other value will not bind.
};

}} // namespace ppl::nn

#endif
