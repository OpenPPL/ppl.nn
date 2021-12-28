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
       vector<dataformat_t> output_formats;
       // fill output_formats
       cuda_engine->Configure(CUDA_CONF_SET_OUTPUT_FORMAT, output_formats.data(), output_formats.size());
       @endcode
    */
    CUDA_CONF_SET_OUTPUT_FORMAT = 0,

    /**
       @brief set output data type

       @note example:
       @code{.cpp}
       vector<datatype_t> output_types;
       // fill output_types;
       cuda_engine->Configure(CUDA_CONF_SET_OUTPUT_TYPE, output_types.data(), output_types.size());
       @endcode
    */
    CUDA_CONF_SET_OUTPUT_TYPE,

    /**
       @brief set default kernel type

       @note example:
       @code{.cpp}
       datatype_t default_kernel_type;
       // fill output_types;
       cuda_engine->Configure(CUDA_CONF_USE_DEFAULT_KERNEL_TYPE, type);
       @endcode
    */
    CUDA_CONF_USE_DEFAULT_KERNEL_TYPE,

    /**
       @brief set init input dims as a hint for graph optimization

       @note example:
       @code{.cpp}
       vector<utils::Array<int64_t>> dims;
       // fill dims of each input
       engine->Configure(CUDA_CONF_SET_INPUT_DIMS, dims.data(), dims.size());
       @endcode
    */
    CUDA_CONF_SET_INPUT_DIMS,

    /**
       @brief use default algorithms for conv and gemm

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_USE_DEFAULT_ALGORITHMS, true/false);
       @endcode
    */
    CUDA_CONF_USE_DEFAULT_ALGORITHMS,

    /**
       @param json_file a json file containing quantization information

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_SET_QUANTIZATION, json_file);
       @endcode
    */
    CUDA_CONF_SET_QUANTIZATION,

    /**
       @param json_file a json file used to store selected algos' index information

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_EXPORT_ALGORITHMS, json_file);
       @endcode
    */
    CUDA_CONF_EXPORT_ALGORITHMS,

    /**
       @param json_file a json file containing selected algos' index information

       @note example:
       @code{.cpp}
       cuda_engine->Configure(CUDA_CONF_IMPORT_ALGORITHMS, json_file);
       @endcode
    */
    CUDA_CONF_IMPORT_ALGORITHMS,

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
