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

#include "ppl/nn/engines/arm/data_converter.h"
#include <cstring>
#include "ppl/nn/common/logger.h"
#include "ppl/kernel/arm_server/common/data_trans.h"
#include "ppl/kernel/arm_server/common/memory.h"
#include "ppl/kernel/arm_server/cast/neon/cast.h"
using namespace ppl::common;
using namespace ppl::kernel::arm_server;

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode ArmDataConverter::Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                               const TensorShape& src_desc) const {
    LOG(DEBUG) << "ARM Data Converter from data format " << src_desc.GetDataFormat() << " to "
               << dst_desc.GetDataFormat();
    LOG(DEBUG) << "ARM Data Converter from data type " << src_desc.GetDataType() << " to " << dst_desc.GetDataType();
    if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() && dst_desc.GetDataType() == src_desc.GetDataType()) {
        return memory_copy(src.addr, src_desc.CalcBytesIncludingPadding(), dst->addr);
    } else if (dst_desc.GetDataFormat() != src_desc.GetDataFormat() &&
               dst_desc.GetDataType() == src_desc.GetDataType()) {
        if (dst_desc.GetDataType() == DATATYPE_FLOAT32) {
            if (dst_desc.GetDataFormat() == DATAFORMAT_N4CX && src_desc.GetDataFormat() == DATAFORMAT_NDARRAY) {
                NdarrayToN4cxFp32((float*)(src.addr), src_desc.GetDim(0), src_desc.GetDim(1), src_desc.GetDim(2),
                                  src_desc.GetDim(3), (float*)(dst->addr));
                return RC_SUCCESS;
            } else if (dst_desc.GetDataFormat() == DATAFORMAT_NDARRAY && src_desc.GetDataFormat() == DATAFORMAT_N4CX) {
                N4cxToNdarrayFp32((float*)(src.addr), src_desc.GetDim(0), src_desc.GetDim(1), src_desc.GetDim(2),
                                  src_desc.GetDim(3), (float*)(dst->addr));
                return RC_SUCCESS;
            }
#ifdef PPLNN_USE_ARMV8_2_FP16
        } else if (dst_desc.GetDataType() == DATATYPE_FLOAT16) {
            if (dst_desc.GetDataFormat() == DATAFORMAT_N8CX && src_desc.GetDataFormat() == DATAFORMAT_NDARRAY) {
                NdarrayToN8cxFp16((__fp16*)(src.addr), src_desc.GetDim(0), src_desc.GetDim(1), src_desc.GetDim(2),
                                  src_desc.GetDim(3), (__fp16*)(dst->addr));
                return RC_SUCCESS;
            } else if (dst_desc.GetDataFormat() == DATAFORMAT_NDARRAY && src_desc.GetDataFormat() == DATAFORMAT_N8CX) {
                N8cxToNdarrayFp16((__fp16*)(src.addr), src_desc.GetDim(0), src_desc.GetDim(1), src_desc.GetDim(2),
                                  src_desc.GetDim(3), (__fp16*)(dst->addr));
                return RC_SUCCESS;
            }
#endif
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
    } else if (dst_desc.GetDataFormat() != src_desc.GetDataFormat() &&
               dst_desc.GetDataType() != src_desc.GetDataType()) {
        if (dst_desc.GetDataType() == DATATYPE_FLOAT32 && dst_desc.GetDataFormat() == DATAFORMAT_NDARRAY &&
            src_desc.GetDataType() == DATATYPE_FLOAT16 && src_desc.GetDataFormat() == DATAFORMAT_N8CX) {
            N8cxFp16ToNdarrayFp32((__fp16*)(src.addr), src_desc.GetDim(0), src_desc.GetDim(1), src_desc.GetDim(2),
                                  src_desc.GetDim(3), (float*)(dst->addr));
            return RC_SUCCESS;
        } else if (dst_desc.GetDataType() == DATATYPE_FLOAT16 && dst_desc.GetDataFormat() == DATAFORMAT_N8CX &&
                   src_desc.GetDataType() == DATATYPE_FLOAT32 && src_desc.GetDataFormat() == DATAFORMAT_NDARRAY) {
            NdarrayFp32ToN8cxFp16((float*)(src.addr), src_desc.GetDim(0), src_desc.GetDim(1), src_desc.GetDim(2),
                                  src_desc.GetDim(3), (__fp16*)(dst->addr));
            return RC_SUCCESS;
        }
    } else if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() &&
               dst_desc.GetDataType() != src_desc.GetDataType()) {
        if (dst_desc.GetDataType() == DATATYPE_FLOAT32 && src_desc.GetDataType() == DATATYPE_FLOAT16) {
            Fp16ToFp32((__fp16*)src.addr, src_desc.CalcElementsIncludingPadding(), (float*)dst->addr);
            return RC_SUCCESS;
        } else if (dst_desc.GetDataType() == DATATYPE_FLOAT16 && src_desc.GetDataType() == DATATYPE_FLOAT32) {
            Fp32ToFp16((float*)src.addr, src_desc.CalcElementsIncludingPadding(), (__fp16*)dst->addr);
            return RC_SUCCESS;
        } else {
            return ppl::kernel::arm_server::neon::cast(&src_desc, &dst_desc, src.addr, dst->addr);
        }
#endif
    }
    LOG(ERROR) << "Invalid data type conversion";
    return RC_UNSUPPORTED;
}

ppl::common::RetCode ArmDataConverter::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                                     const TensorShape& src_desc) const {
    BufferDesc dst_wrapper(dst);
    return Convert(&dst_wrapper, dst_desc, src, src_desc);
}

ppl::common::RetCode ArmDataConverter::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                                       const TensorShape& src_desc) const {
    return Convert(dst, dst_desc, BufferDesc(const_cast<void*>(src)), src_desc);
}

}}} // namespace ppl::nn::arm
