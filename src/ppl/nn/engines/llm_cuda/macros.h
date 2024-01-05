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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_MACROS_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_MACROS_H_

#include "ppl/nn/common/logger.h"
#include "ppl/common/stripfilename.h"

#ifndef PPLNN_LLM_CUDA_DEBUG_TRACE
#if defined(DEBUG) || !defined(NDEBUG)
#include <stdio.h>
#define PPLNN_LLM_CUDA_DEBUG_TRACE(fmt, ...) \
    do { \
        std::string __filename(__FILE__); \
        auto __beg = __filename.rfind("kernels/") + 8; \
        fprintf(stderr, "[LLMCUDA][%s:%d] " fmt, __filename.substr(__beg, std::string::npos).c_str(), __LINE__, ##__VA_ARGS__); \
        fflush(stderr); \
    } while (0)
#else
#define PPLNN_LLM_CUDA_DEBUG_TRACE(fmt, ...)
#endif // DEBUG
#endif // Not define PPLNN_LLM_CUDA_DEBUG_TRACE

#if defined(DEBUG) || !defined(NDEBUG)
#define PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(X)                                                                        \
    do {                                                                                                         \
        if (X->GetShape()->IsScalar()) {                                                                          \
            PPLNN_LLM_CUDA_DEBUG_TRACE(" ScalarName: [%s]\n", X->GetName());                                            \
            PPLNN_LLM_CUDA_DEBUG_TRACE(" |-Data: %p\n", X->GetBufferPtr());                                          \
        } else {                                                                                                 \
            PPLNN_LLM_CUDA_DEBUG_TRACE(" TensorName: [%s]\n", X->GetName());                                            \
            PPLNN_LLM_CUDA_DEBUG_TRACE(" |-Data: %p\n", X->GetBufferPtr());                                            \
            PPLNN_LLM_CUDA_DEBUG_TRACE(" |-DimCount: %u\n", X->GetShape()->GetDimCount());                                \
            for (uint32_t __idx = 0; __idx < X->GetShape()->GetDimCount(); ++__idx) {                                         \
                PPLNN_LLM_CUDA_DEBUG_TRACE(" |-  Dim[%u]: %ld\tPads: [%hu, %hu]\n", __idx, X->GetShape()->GetDim(__idx),          \
                                      X->GetShape()->GetPadding0(__idx), X->GetShape()->GetPadding1(__idx));               \
            }                                                                                                    \
        }                                                                                                        \
        PPLNN_LLM_CUDA_DEBUG_TRACE(" |-DataType: %s\n", ppl::common::GetDataTypeStr(X->GetShape()->GetDataType()));       \
        PPLNN_LLM_CUDA_DEBUG_TRACE(" |-DataFormat: %s\n", ppl::common::GetDataFormatStr(X->GetShape()->GetDataFormat())); \
        const uint64_t __num_elem = X->GetShape()->CalcElementsIncludingPadding();                                          \
        if (X->GetShape()->GetDataType() == ppl::common::DATATYPE_INT64 && __num_elem <= 32 && __num_elem > 0) {    \
            PPLNN_LLM_CUDA_DEBUG_TRACE(" |-Value(s):\n");                                                                  \
            std::vector<int64_t> __values(__num_elem);                                   \
            auto __status = X->CopyToHost(__values.data());                                                                          \
            if (ppl::common::RC_SUCCESS != __status) {                                                                   \
                LOG(ERROR) << "CopyToHost of tensor[" << X->GetName() << "] failed: " << ppl::common::GetRetCodeStr(__status);\
                return __status;                                                                                           \
            }                                                                                                               \
            std::string __values_str;                                                                                   \
            for (uint32_t __idx = 0; __idx < __num_elem; ++__idx) {                                                             \
                __values_str += std::to_string(__values[__idx]) + " ";                                                      \
                if ((__idx + 1) % 8 == 0) {                                                                      \
                    PPLNN_LLM_CUDA_DEBUG_TRACE(" |-  %s\n", __values_str.c_str());                                          \
                    __values_str.clear();                                                                       \
                }                                                                                                       \
            }                                                                                                       \
            if (!__values_str.empty()) {                                                                                    \
                PPLNN_LLM_CUDA_DEBUG_TRACE(" |-  %s\n", __values_str.c_str());                                                       \
            }                                                                                                               \
        }                                                                                                                           \
    } while (0)
#else
#define PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(X) do {} while (0)
#endif

#define PPLNN_LLM_CUDA_REQUIRED_INPUT(X, IDX) \
    auto X = ctx->GetInputCount() > (IDX) ? ctx->GetInput<TensorImpl>(IDX) : nullptr; \
    if (!X) { \
        LOG(ERROR) << "Input \""<< #X << "\" is required."; \
        return ppl::common::RC_NOT_FOUND; \
    } do {} while (0)

#define PPLNN_LLM_CUDA_OPTIONAL_INPUT(X, IDX) \
    auto X = ctx->GetInputCount() > (IDX) ? ctx->GetInput<TensorImpl>(IDX) : nullptr

#define PPLNN_LLM_CUDA_REQUIRED_OUTPUT(X, IDX) \
    auto X = ctx->GetOutputCount() > (IDX) ? ctx->GetOutput<TensorImpl>(IDX) : nullptr; \
    if (!X) { \
        LOG(ERROR) << "Output \""<< #X << "\" is required."; \
        return ppl::common::RC_NOT_FOUND; \
    } do {} while (0)

#define PPLNN_LLM_CUDA_OPTIONAL_OUTPUT(X, IDX) \
    auto X = ctx->GetOutputCount() > (IDX) ? ctx->GetOutput<TensorImpl>(IDX) : nullptr

#define PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(X) \
    do {\
        X->SetDevice(GetCudaDevice());\
        auto __status = X->ReallocBuffer();\
        if (__status != ppl::common::RC_SUCCESS) {\
            LOG(ERROR) << "ReallocBuffer for tensor[" << X->GetName() << "] failed: " << ppl::common::GetRetCodeStr(__status);\
            return __status;\
        }\
    } while (0)

#define PPLNN_LLM_CUDA_RESHAPE_OUTPUTS() \
    do {\
        auto rc = Reshape(ctx);\
        if (ppl::common::RC_SUCCESS != rc) {\
            LOG(ERROR) << "Reshape kernel[" << GetName() << "] failed: " << ppl::common::GetRetCodeStr(rc);\
            return rc;\
        }\
    } while (0)

#endif // _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_MACROS_H_
