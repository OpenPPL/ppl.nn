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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIONS_H_

namespace ppl { namespace nn { namespace arm {

enum {
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

/** @brief graph optimization level */
enum {
    /** disable all graph optimization */
    OPT_DISABLE_ALL = 0,

    /** enable basic(level0) graph optimization */
    OPT_ENABLE_BASIC = 1,

    /** enable extended(level0 & level1) graph optimization */
    OPT_ENABLE_EXTENDED = 2,

    /** enable all(level0 ~ level2) graph optimization */
    OPT_ENABLE_ALL = 3,
};

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
};

/** @brief dynamic tuning level */
enum {
    /** turn off dynamic tuning */
    TUNING_OFF = 0,

    /** use dynamic tuning to select algorithm */
    TUNING_SELECT_ALGO = 1,

    /** use dynamic tuning to select algorithm with blocking size */
    TUNING_SELECT_BLK_SIZE = 2,
};

/** @brief options for arm::DeviceContext::Configure() */
enum {
    DEV_CONF_MAX,
};

}}} // namespace ppl::nn::arm

#endif
