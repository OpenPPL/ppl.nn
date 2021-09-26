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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_OPTIONS_H_

namespace ppl { namespace nn {

enum {
    /**
       @brief set output data format

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_OUTPUT_FORMAT, DATAFORMAT_NDARRAY);
       @endcode
    */
    CUDA_CONF_SET_OUTPUT_FORMAT = 0,

    /**
       @brief set output data type

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_OUTPUT_TYPE, DATATYPE_FLOAT32);
       @endcode
    */
    CUDA_CONF_SET_OUTPUT_TYPE,

    /**
       @brief set init input dims for compiler

       @note example:
       @code{.cpp}
       std::string dims = "1_3_224_224";
       engine->Configure(CUDA_CONF_SET_COMPILER_INPUT_SHAPE, dims.c_str());
       @endcode
    */
    CUDA_CONF_SET_COMPILER_INPUT_SHAPE,

    /**
       @brief use default algorithms for conv and gemm

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_USE_DEFAULT_ALGORITHMS, true/false);
       @endcode
    */
    CUDA_CONF_USE_DEFAULT_ALGORITHMS,

    /**
       @brief the name of json file that saves quantization information

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_QUANTIZATION, "quantization.json");
       @endcode
    */
    CUDA_CONF_SET_QUANTIZATION,
   
    /**
       @brief the name of json file that saves selected algos' index information

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_ALGORITHM, "algo_info.json");
       @endcode
    */
    CUDA_CONF_SET_ALGORITHM,
    
    /** max value */
    CUDA_CONF_MAX,
};

/** @brief memory management policies */
enum {
    /** less memory usage, does not support vGPU now */
    CUDA_MM_COMPACT = 0,

    /** best fit first, will use more memory */
    CUDA_MM_BEST_FIT = 1,
};

}} // namespace ppl::nn

#endif
