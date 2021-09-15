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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_X86_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_X86_OPTIONS_H_

namespace ppl { namespace nn {

enum {
    /**
       @brief disable avx512 support

       @note example:
       @code{.cpp}
       x86_engine->Configure(X86_CONF_DISABLE_AVX512);
       @endcode
    */
    X86_CONF_DISABLE_AVX512 = 0,

    /**
       @brief disable avx, fma3 and avx512 support

       @note example:
       @code{.cpp}
       x86_engine->Configure(X86_CONF_DISABLE_AVX_FMA3);
       @endcode
    */
    X86_CONF_DISABLE_AVX_FMA3 = 1,

    /** max value */
    X86_CONF_MAX,
};

/** @brief memory management policies */
enum {
    /** less memory usage, may cause performance loss */
    X86_MM_COMPACT = 0,

    /** most recently used first, will use more memory */
    X86_MM_MRU = 1,
};

}} // namespace ppl::nn

#endif
