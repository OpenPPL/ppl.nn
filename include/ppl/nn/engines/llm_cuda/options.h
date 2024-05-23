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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPTIONS_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_OPTIONS_H_

namespace ppl { namespace nn { namespace llm { namespace cuda {

/** @brief engine configuration options */
enum {
    /**
       @param tensor_parallel_nccl_comm (ncclComm_t) tensor parallel nccl comm handle

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_SET_TP_NCCL_COMM, tensor_parallel_nccl_comm);
       @endcode
    */
    ENGINE_CONF_SET_TP_NCCL_COMM = 0,

    /**
       @brief uint32_t, set graph kernel fusion on(1)/off(0), default is on

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_GRAPH_FUSION, uint32_t);
       @endcode
    */
    ENGINE_CONF_GRAPH_FUSION = 1,

    /**
       @brief uint32_t, set dump tensors' data on(1)/off(0), default is off

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_TENSOR_DEBUG, uint32_t);
       @endcode
    */
    ENGINE_CONF_TENSOR_DEBUG = 2,

    /**
       @brief const char*, directory to save dumped tensors' data, default is "."

       @note example:
       @code{.cpp}
       cuda_engine->Configure(ENGINE_CONF_DEBUG_DATA_DIR, const char*);
       @endcode
    */
    ENGINE_CONF_DEBUG_DATA_DIR = 3,

    ENGINE_CONF_MAX,
};

/** @brief memory management policies */
enum {
    /** naive implementation */
    MM_PLAIN,

    /** less memory usage, vGPU not supported now */
    MM_COMPACT,

    /** bestfit among free chunks */
    MM_BEST_FIT,
};

/** @brief quantization method */
enum {
    /** no quantize */
    QUANT_METHOD_NONE,

    /** online quantization, int8 tensor and int8 weight */
    QUANT_METHOD_ONLINE_I8I8,

    /** online quantization, fp16 tensor and int4 weight */
    QUANT_METHOD_ONLINE_I4F16,
};

/** @brief cublas layout hint, currently for selecting matrix layout for int8 gemm */
enum {
    /** use default matrix layout */
    CUBLAS_LAYOUT_DEFAULT,

    /** use ampere specify matrix layout
     * NOTE: It may improve performance on some device(orin),
     *       but also may lost performance on some device(A100).
     *       The actual performance is subject to testing.
     */
    CUBLAS_LAYOUT_AMPERE,
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

    /**
       @note example:
       @code{.cpp}
       cudaStream_t stream = 0;
       cuda_engine->Configure(DEV_CONF_GET_STREAM, &stream);
       @endcode
    */
    DEV_CONF_GET_STREAM,

    DEV_CONF_MAX,
};

}}}} // namespace ppl::nn::llm::cuda

#endif
