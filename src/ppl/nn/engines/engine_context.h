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

#ifndef _ST_HPC_PPL_NN_ENGINES_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_ENGINES_ENGINE_CONTEXT_H_

#include "ppl/nn/common/device.h"

namespace ppl { namespace nn {

/**
   @class EngineContext
   @brief resources needed by a `Runtime` instance
   @note Each `Runtime` has only one `EngineContext` and an `EngineContext`
   is used only by one `Runtime` instance.
*/
class EngineContext {
public:
    virtual ~EngineContext() {}

    /** @brief create a `Device` instance used by this `Runtime` */
    virtual Device* CreateDevice() = 0;
};

}} // namespace ppl::nn

#endif
