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

#include "ppl/kernel/arm_server/common/pad_channel.h"

namespace ppl { namespace kernel { namespace arm_server {

template <typename eT, int32_t c_blk>
inline void pad_channel_zero_common_kernel(
    const int64_t length, 
    const int64_t c_eff, 
    eT* data) {

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < length; i++) {
        for (int64_t c = c_eff; c < c_blk; c++) {
            data[i * c_blk + c] = 0;
        }
    }
}

template <typename eT, int32_t c_blk, int32_t c_eff>
inline void pad_channel_zero_template_kernel(
    const int64_t length, 
    eT* data) {

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < length; i++) {
        if (c_eff <= 0 && c_blk > 0) data[i * c_blk + 0] = 0;
        if (c_eff <= 1 && c_blk > 1) data[i * c_blk + 1] = 0;
        if (c_eff <= 2 && c_blk > 2) data[i * c_blk + 2] = 0;
        if (c_eff <= 3 && c_blk > 3) data[i * c_blk + 3] = 0;
        if (c_eff <= 4 && c_blk > 4) data[i * c_blk + 4] = 0;
        if (c_eff <= 5 && c_blk > 5) data[i * c_blk + 5] = 0;
        if (c_eff <= 6 && c_blk > 6) data[i * c_blk + 6] = 0;
        if (c_eff <= 7 && c_blk > 7) data[i * c_blk + 7] = 0;
    }
}

template <typename eT, int32_t c_blk>
static ppl::common::RetCode pad_channel_zero_common(
    const int64_t outer_dims, 
    const int64_t channels, 
    const int64_t inner_dims, 
    eT* data) {

    const int64_t pad_c = round_up(channels, c_blk);
    if (channels == pad_c) {
        return ppl::common::RC_SUCCESS;
    }

    const int64_t c_round = round(channels, c_blk);
    const int64_t c_eff = channels - c_round;

    for (int64_t od = 0; od < outer_dims; od++) {
        eT* p_data = data + (od * pad_c + c_round) * inner_dims;
        switch (c_eff) {
            case 0: pad_channel_zero_template_kernel<eT, c_blk, 0>(inner_dims, p_data); break;
            case 1: pad_channel_zero_template_kernel<eT, c_blk, 1>(inner_dims, p_data); break;
            case 2: pad_channel_zero_template_kernel<eT, c_blk, 2>(inner_dims, p_data); break;
            case 3: pad_channel_zero_template_kernel<eT, c_blk, 3>(inner_dims, p_data); break;
            case 4: pad_channel_zero_template_kernel<eT, c_blk, 4>(inner_dims, p_data); break;
            case 5: pad_channel_zero_template_kernel<eT, c_blk, 5>(inner_dims, p_data); break;
            case 6: pad_channel_zero_template_kernel<eT, c_blk, 6>(inner_dims, p_data); break;
            case 7: pad_channel_zero_template_kernel<eT, c_blk, 7>(inner_dims, p_data); break;
            default: pad_channel_zero_common_kernel<eT, c_blk>(inner_dims, c_eff, p_data); break;
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline int64_t get_c_blk(const ppl::common::dataformat_t data_format) {
    switch (data_format) {
        case ppl::common::DATAFORMAT_N2CX: return 2;
        case ppl::common::DATAFORMAT_N4CX: return 4;
        case ppl::common::DATAFORMAT_N8CX: return 8;
        case ppl::common::DATAFORMAT_N16CX: return 16;
        default: return 1;
    }
}

template <int32_t c_blk>
inline ppl::common::RetCode pad_channel_zero_wrapper(
    const ppl::common::datatype_t data_type, 
    const int64_t outer_dims, 
    const int64_t channels, 
    const int64_t inner_dims, 
    void* data) {

    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return pad_channel_zero_common<uint8_t, c_blk>(outer_dims, channels, inner_dims, (uint8_t*)data);
        case 2: return pad_channel_zero_common<uint16_t, c_blk>(outer_dims, channels, inner_dims, (uint16_t*)data);
        case 4: return pad_channel_zero_common<uint32_t, c_blk>(outer_dims, channels, inner_dims, (uint32_t*)data);
        case 8: return pad_channel_zero_common<uint64_t, c_blk>(outer_dims, channels, inner_dims, (uint64_t*)data);
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode pad_channel_zero(
    const ppl::common::TensorShape* shape, 
    void* data) {

    const auto data_type = shape->GetDataType();
    const auto data_format = shape->GetDataFormat();
    const int64_t c_blk = get_c_blk(data_format);

    if (c_blk <= 1) {
        return ppl::common::RC_SUCCESS;
    }

    const int64_t dim_count = shape->GetDimCount();
    if (dim_count < 3) {    // NBCX must have at least 3 dims
        return ppl::common::RC_INVALID_VALUE;
    }

    const int64_t c_dim_idx = 1;
    const int64_t channels = shape->GetDim(c_dim_idx);
    const int64_t pad_c = round_up(channels, c_blk);
    if (channels == pad_c) {
        return ppl::common::RC_SUCCESS;
    }

    int64_t outer_dims = 1;
    int64_t inner_dims = 1;
    for (int64_t i = 0; i < c_dim_idx; i++) {
        outer_dims *= shape->GetDim(i);
    }
    for (int64_t i = c_dim_idx + 1; i < dim_count; i++) {
        inner_dims *= shape->GetDim(i);
    }

    switch (c_blk) {
        case 2: return pad_channel_zero_wrapper<2>(data_type, outer_dims, channels, inner_dims, data);
        case 4: return pad_channel_zero_wrapper<4>(data_type, outer_dims, channels, inner_dims, data);
        case 8: return pad_channel_zero_wrapper<8>(data_type, outer_dims, channels, inner_dims, data);
        case 16: return pad_channel_zero_wrapper<16>(data_type, outer_dims, channels, inner_dims, data);
        default: break;
    }

    return ppl::common::RC_UNSUPPORTED;
}


}}}; // namespace ppl::kernel::arm_server
