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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_MACROS_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_MACROS_H_

#include "ppl/nn/common/logger.h"
#include "ppl/common/stripfilename.h"

#ifndef PPLNN_X86_DEBUG_TRACE
#if defined(DEBUG) || !defined(NDEBUG)
#include <stdio.h>
#define PPLNN_X86_DEBUG_TRACE(fmt, ...) \
    fprintf(stderr, "T [%s:%d] " fmt, ppl::common::stripfilename(__FILE__), __LINE__, ##__VA_ARGS__)
#else
#define PPLNN_X86_DEBUG_TRACE(fmt, ...)
#endif // DEBUG
#endif // Not define PPLNN_X86_DEBUG_TRACE

#define PPL_X86_TENSOR_PRINT_DEBUG_MSG(X)                                                                        \
    do {                                                                                                         \
        if (X->GetShape().IsScalar()) {                                                                          \
            PPLNN_X86_DEBUG_TRACE("Scalar name: %s\n", X->GetName());                                            \
            PPLNN_X86_DEBUG_TRACE("\tdata: %p type: %u\n", X->GetBufferPtr(), X->GetShape().GetDataType());      \
        } else {                                                                                                 \
            PPLNN_X86_DEBUG_TRACE("Tensor name: %s\n", X->GetName());                                            \
            PPLNN_X86_DEBUG_TRACE("\tdata: %p\n", X->GetBufferPtr());                                            \
            PPLNN_X86_DEBUG_TRACE("DimCount: %u\n", X->GetShape().GetDimCount());                                \
            for (uint32_t i = 0; i < X->GetShape().GetDimCount(); ++i) {                                         \
                PPLNN_X86_DEBUG_TRACE("\tdim[%u]: %ld\tpads: [%hu, %hu]\n", i, X->GetShape().GetDim(i),          \
                                      X->GetShape().GetPadding0(i), X->GetShape().GetPadding1(i));               \
            }                                                                                                    \
        }                                                                                                        \
        PPLNN_X86_DEBUG_TRACE("DataType: %s\n", ppl::common::GetDataTypeStr(X->GetShape().GetDataType()));       \
        PPLNN_X86_DEBUG_TRACE("DataFormat: %s\n", ppl::common::GetDataFormatStr(X->GetShape().GetDataFormat())); \
    } while (0)

#define PPLNN_X86_REQUIRED_INPUT(X, IDX) \
    auto X = ctx->GetInputCount() > IDX ? ctx->GetInput<TensorImpl>(IDX) : nullptr; \
    if (!X) { \
        LOG(ERROR) << "Input \""<< #X << "\" is required."; \
        return ppl::common::RC_NOT_FOUND; \
    } do {} while (0)

#define PPLNN_X86_OPTIONAL_INPUT(X, IDX) \
    auto X = ctx->GetInputCount() > IDX ? ctx->GetInput<TensorImpl>(IDX) : nullptr

#define PPLNN_X86_REQUIRED_OUTPUT(X, IDX) \
    auto X = ctx->GetOutputCount() > IDX ? ctx->GetOutput<TensorImpl>(IDX) : nullptr; \
    if (!X) { \
        LOG(ERROR) << "Output \""<< #X << "\" is required."; \
        return ppl::common::RC_NOT_FOUND; \
    } do {} while (0)

#define PPLNN_X86_OPTIONAL_OUTPUT(X, IDX) \
    auto X = ctx->GetOutputCount() > IDX ? ctx->GetOutput<TensorImpl>(IDX) : nullptr

#endif
