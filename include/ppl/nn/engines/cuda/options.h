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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIONS_H_

namespace ppl { namespace nn { namespace cuda {

enum {
    /**
       @brief set kernel type

       @note example:
       @code{.cpp}
       datatype_t kernel_type;
       cuda_engine->Configure(ENGINE_CONF_SET_KERNEL_TYPE, kernel_type);
       @endcode
    */
    ENGINE_CONF_SET_KERNEL_TYPE,

    /**
       @brief set init input dims as a hint for graph optimization

       @note example:
       @code{.cpp}
       vector<utils::Array<int64_t>> dims;
       // fill dims of each input
       engine->Configure(ENGINE_CONF_SET_INPUT_DIMS, dims.data(), dims.size());
       @endcode
    */
    ENGINE_CONF_SET_INPUT_DIMS,

    /**
       @brief use default algorithms for conv and gemm

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_USE_DEFAULT_ALGORITHMS, true/false);
       @endcode
    */
    ENGINE_CONF_USE_DEFAULT_ALGORITHMS,

    /**
       @param json_buf a json buffer containing quantization information

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_SET_QUANT_INFO, json_buf, json_size);
       @endcode
    */
    ENGINE_CONF_SET_QUANT_INFO,

    /**
       @brief sets the callback function and arg for exporting algorithms info:
       cuda_engine->Configure(ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER, callback, arg);

       @note this call just sets the callback function and arg, it does not call the function immediately

       @param callback a C-style callback function `void (*)(const char* data, uint64_t bytes, void* arg)`
       @param arg a pointer that is passed to `callback`

       @note example:
       @code{.cpp}
       static void SaveAlgoInfo(const char* data, uint64_t bytes, void* arg) {
           auto content = (string*)arg;
           content->assign(data, bytes);
       }
       string content;
       cuda_engine->Configure(ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER, SaveAlgoInfo, &content);
       @endcode
    */
    ENGINE_CONF_SET_EXPORT_ALGORITHMS_HANDLER,

    /**
       @param json_buffer pointer to a json buffer containing selected algos' index information
       @param buffer_size length of the buffer

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER, json_buffer, buffer_size);
       @endcode
    */
    ENGINE_CONF_IMPORT_ALGORITHMS_FROM_BUFFER,

      /**
       @param torch2onnx torch_name --> onnx_name mapping
       @param name2val torch_name --> data_ptr mapping

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_REFIT_CONSTANT_WEIGHTS, torch2onnx, name2val);
       @endcode
    */
    ENGINE_CONF_REFIT_CONSTANT_WEIGHTS,

    /** max value */
    ENGINE_CONF_MAX,
};

/** @brief memory management policies */
enum {
    /** less memory usage, does not support vGPU now */
    MM_COMPACT = 0,

    /** best fit first, will use more memory */
    MM_BEST_FIT = 1,

    /** plain implementation */
    MM_PLAIN = 2,
};

/** @brief device configuration options */
enum {
    /**
       @note example:
       @code{.cpp}
       int device_id = 0;
       cuda_engine->Configure(DEV_CONF_GET_DEVICE_ID, &device_id);
       @endcode
    */
    DEV_CONF_GET_DEVICE_ID,

    DEV_CONF_MAX,
};

}}} // namespace ppl::nn::cuda

#endif
