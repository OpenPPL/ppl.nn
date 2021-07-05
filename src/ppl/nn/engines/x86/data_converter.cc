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

#include <cstring> // memcpy
#include <vector>

#include "ppl/nn/engines/x86/data_converter.h"
#include "ppl/kernel/x86/common/cast.h"
#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/int64/reorder.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode X86DataConverter::Convert(BufferDesc* dst_buf, const TensorShape& dst_desc, const BufferDesc& src_buf,
                                  const TensorShape& src_desc) const {
    const auto src_data_type = src_desc.GetDataType();
    const auto dst_data_type = dst_desc.GetDataType();
    const auto src_data_format = src_desc.GetDataFormat();
    const auto dst_data_format = dst_desc.GetDataFormat();

    if (dst_data_format == src_data_format && dst_data_type == src_data_type) {
        memcpy(dst_buf->addr, src_buf.addr, src_desc.GetBytesIncludingPadding());
        return RC_SUCCESS;
    } else if (dst_data_format == src_data_format && dst_data_type != src_data_type) {
        return ppl::kernel::x86::cast(&src_desc, &dst_desc, src_buf.addr, dst_buf->addr);
    } else if (dst_data_format != src_data_format && dst_data_type == src_data_type) {
        if (GetSizeOfDataType(dst_data_type) == 4) {
            if (dst_data_format == DATAFORMAT_N16CX && src_data_format == DATAFORMAT_NDARRAY) {
                if (MayUseISA(ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_fp32_avx(&src_desc, (const float*)(src_buf.addr),
                                                                            (float*)(dst_buf->addr));
                } else {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_fp32(&src_desc, (const float*)(src_buf.addr),
                                                                        (float*)(dst_buf->addr));
                }
            } else if (dst_data_format == DATAFORMAT_NDARRAY && src_data_format == DATAFORMAT_N16CX) {
                if (MayUseISA(ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_fp32_avx(&src_desc, (const float*)(src_buf.addr),
                                                                            (float*)(dst_buf->addr));
                } else {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_fp32(&src_desc, (const float*)(src_buf.addr),
                                                                        (float*)(dst_buf->addr));
                }
            }
        } else if (GetSizeOfDataType(dst_data_type) == 8) {
            if (dst_data_format == DATAFORMAT_N16CX && src_data_format == DATAFORMAT_NDARRAY) {
                if (MayUseISA(ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_int64_avx(&src_desc, (const int64_t*)(src_buf.addr),
                                                                             (int64_t*)(dst_buf->addr));
                } else {
                    return ppl::kernel::x86::reorder_ndarray_n16cx_int64(&src_desc, (const int64_t*)(src_buf.addr),
                                                                         (int64_t*)(dst_buf->addr));
                }
            } else if (dst_data_format == DATAFORMAT_NDARRAY && src_data_format == DATAFORMAT_N16CX) {
                if (MayUseISA(ISA_X86_AVX)) {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_int64_avx(&src_desc, (const int64_t*)(src_buf.addr),
                                                                             (int64_t*)(dst_buf->addr));
                } else {
                    return ppl::kernel::x86::reorder_n16cx_ndarray_int64(&src_desc, (const int64_t*)(src_buf.addr),
                                                                         (int64_t*)(dst_buf->addr));
                }
            }
        }
    } else { // dst_data_format != src_data_format && dst_data_type != src_data_type
        std::vector<uint8_t> temp_buf_vec(src_desc.GetBytesIncludingPadding());
        auto temp_desc = dst_desc;
        temp_desc.SetDataFormat(src_data_format);
        BufferDesc temp_buf(temp_buf_vec.data());
        auto status = Convert(&temp_buf, temp_desc, src_buf, src_desc);
        if (status != RC_SUCCESS) {
            return status;
        }
        return Convert(dst_buf, dst_desc, temp_buf, temp_desc);
    }

    return RC_UNSUPPORTED;
}

RetCode X86DataConverter::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                        const TensorShape& src_desc) const {
    BufferDesc dst_wrapper(dst);
    return Convert(&dst_wrapper, dst_desc, src, src_desc);
}

RetCode X86DataConverter::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                          const TensorShape& src_desc) const {
    return Convert(dst, dst_desc, BufferDesc(const_cast<void*>(src)), src_desc);
}

}}} // namespace ppl::nn::x86
