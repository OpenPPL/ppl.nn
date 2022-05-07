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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIONS_H_

namespace ppl { namespace nn { namespace riscv {

/** @brief winograd level */
enum {
    /** turn off winograd */
    WG_OFF = 0,

    /** use winograd and select block size automatically */
    WG_ON = 1,

    /** use winograd blk2 if possible */
    WG_ON_B2 = 2,

    /** use wingorad blk4 if possible */
    WG_ON_B4 = 3,

    /** use wingorad blk6 if possible */
    WG_ON_B6 = 4,
};

/** @brief memory management policies */
enum {
    /** less memory usage, may cause performance loss */
    MM_COMPACT = 0,

    /** most recently used first, will use more memory */
    MM_MRU = 1,
};

/** @brief options for riscv::DeviceContext::Configure() */
enum {
    DEV_CONF_MAX,
};

}}} // namespace ppl::nn::riscv

#endif
