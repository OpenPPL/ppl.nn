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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIONS_H_

namespace ppl { namespace nn { namespace x86 {

/** @brief options for X86Engine::Configure() */
enum {
    /**
       @brief uint32_t, set graph kernel fusion on(1)/off(0), default is on

       @note example:
       @code{.cpp}
       x86_engine->Configure(ENGINE_CONF_GRAPH_FUSION, int32_t);
       @endcode
    */
    ENGINE_CONF_GRAPH_FUSION = 0,

    /**
       @brief uint32_t, set dump tensors' data on(1)/off(0), default is off

       @note example:
       @code{.cpp}
       x86_engine->Configure(ENGINE_CONF_TENSOR_DEBUG, int32_t);
       @endcode
    */
    ENGINE_CONF_TENSOR_DEBUG = 1,

    /**
       @brief const char*, directory to save dumped tensors' data, default is "."

       @note example:
       @code{.cpp}
       x86_engine->Configure(ENGINE_CONF_DEBUG_DATA_DIR, const char*);
       @endcode
    */
    ENGINE_CONF_DEBUG_DATA_DIR = 2,

    /** max value */
    ENGINE_CONF_MAX,
};

/** @brief memory management policies */
enum {
    /** less memory usage, may cause performance loss */
    MM_COMPACT = 0,

    /** most recently used first, will use more memory */
    MM_MRU = 1,

    /** plain implementation */
    MM_PLAIN = 2,
};

/** @brief options for x86::DeviceContext::Configure() */
enum {
    DEV_CONF_MAX,
};

}}} // namespace ppl::nn::x86

#endif
