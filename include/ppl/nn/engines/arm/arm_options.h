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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_ARM_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_ARM_OPTIONS_H_

namespace ppl { namespace nn {

enum {
    /** max value */
    ARM_CONF_MAX,
};

/** @brief memory management policies */
enum {
    /** less memory usage, may cause performance loss */
    ARM_MM_COMPACT = 0,

    /** most recently used first, will use more memory */
    ARM_MM_MRU = 1,
};

/** @brief graph optimization level */
enum {
    /** disable all graph optimization */
    ARM_OPT_DISABLE_ALL = 0,

    /** enable basic(level0) graph optimization */
    ARM_OPT_ENABLE_BASIC = 1,

    /** enable extended(level0 & level1) graph optimization */
    ARM_OPT_ENABLE_EXTENDED = 2,

    /** enable all(level0 ~ level2) graph optimization */
    ARM_OPT_ENABLE_ALL = 3,
};

/** @brief winograd level */
enum {
    /** turn off winograd */
    ARM_WG_OFF = 0,

    /** use winograd and select block size automatically */
    ARM_WG_ON = 1,

    /** use winograd blk2 if possible */
    ARM_WG_ON_B2 = 2,

    /** use wingorad blk4 if possible */
    ARM_WG_ON_B4 = 3,
};

/** @brief dynamic tuning level */
enum {
    /** turn off dynamic tuning */
    ARM_TUNING_OFF = 0,

    /** use dynamic tuning to select algorithm */
    ARM_TUNING_SELECT_ALGO = 1,

    /** use dynamic tuning to select algorithm with blocking size */
    ARM_TUNING_SELECT_BLK_SIZE = 2,
};

/** @brief options for arm::DeviceContext::Configure() */
enum {
    /** @brief memory defragmentation. make sure that device is not used when performing defragmentations. */
    ARM_DEV_CONF_MEM_DEFRAG = 0,

    ARM_DEV_CONF_MAX,
};
}} // namespace ppl::nn

#endif
