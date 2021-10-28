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

#include "cudakernel/arithmetic/logical.h"
#include "ppl/common/types.h"
#include <algorithm>

enum LogicalOpType {
    Logical_Unknown = 0,
    Logical_And,
    Logical_OpNum,
    Logical_ForceWord = INT_MAX,
};

struct bool8_ {
    bool x0;
    bool y0;
    bool z0;
    bool w0;
    bool x1;
    bool y1;
    bool z1;
    bool w1;
};

template <LogicalOpType op_type>
__device__ inline bool ppl_logical_scalar(bool a, bool b);

template <>
__device__ inline bool ppl_logical_scalar<Logical_And>(bool a, bool b)
{
    return a && b;
}

template <LogicalOpType op_type>
static __device__ inline bool ppl_logical_vector(bool a, bool b)
{
    bool res;
    res = ppl_logical_scalar<op_type>(a, b);
    return res;
}

template <LogicalOpType op_type>
static __device__ inline bool8_ ppl_logical_vector(bool8_ a, bool8_ b)
{
    bool8_ res;
    res.x0 = ppl_logical_scalar<op_type>(a.x0, b.x0);
    res.y0 = ppl_logical_scalar<op_type>(a.y0, b.y0);
    res.z0 = ppl_logical_scalar<op_type>(a.z0, b.z0);
    res.w0 = ppl_logical_scalar<op_type>(a.w0, b.w0);
    res.x1 = ppl_logical_scalar<op_type>(a.x1, b.x1);
    res.y1 = ppl_logical_scalar<op_type>(a.y1, b.y1);
    res.z1 = ppl_logical_scalar<op_type>(a.z1, b.z1);
    res.w1 = ppl_logical_scalar<op_type>(a.w1, b.w1);
    return res;
}

void ppl_logical_prepare_strides(
    const ppl::nn::TensorShape* tensor_shape0,
    const ppl::nn::TensorShape* tensor_shape1,
    const ppl::nn::TensorShape* tensor_shape_out,
    const int packed_channel,
    uint32_t* stride_in0,
    uint32_t* stride_in1,
    uint32_t* stride_out)
{
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    int max_dims                           = tensor_shape_out->GetDimCount();
    if (pad_tensor_shape0.GetDimCount() < pad_tensor_shape1.GetDimCount()) {
        pad_tensor_shape0.SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape0->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape0.SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape0.SetDim(i, tensor_shape0->GetDim(i - offset));
        }
    } else {
        pad_tensor_shape1.SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape1->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape1.SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape1.SetDim(i, tensor_shape1->GetDim(i - offset));
        }
    }

    const int dimCount   = tensor_shape_out->GetDimCount();
    uint32_t stride0     = 1;
    uint32_t stride1     = 1;
    uint32_t stride_out0 = 1;

    for (int i = dimCount - 1; i >= 0; i--) {
        stride_in0[i] = pad_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        stride_in1[i] = pad_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        stride_out[i] = stride_out0;
        if (i == 1) { // for channel dim, div packed_channel
            stride0 *= (pad_tensor_shape0.GetDim(i) + packed_channel - 1) / packed_channel;
            stride1 *= (pad_tensor_shape1.GetDim(i) + packed_channel - 1) / packed_channel;
            stride_out0 *= (tensor_shape_out->GetDim(i) + packed_channel - 1) / packed_channel;
        } else {
            stride0 *= pad_tensor_shape0.GetDim(i);
            stride1 *= pad_tensor_shape1.GetDim(i);
            stride_out0 *= tensor_shape_out->GetDim(i);
        }
    }
}

void ppl_logical_prepare_strides_nhwc(
    const ppl::nn::TensorShape* tensor_shape0,
    const ppl::nn::TensorShape* tensor_shape1,
    const ppl::nn::TensorShape* tensor_shape_out,
    const int packed_channel,
    uint32_t* stride_in0,
    uint32_t* stride_in1,
    uint32_t* stride_out)
{
    if (tensor_shape0->GetDimCount() < 2 || tensor_shape1->GetDimCount() < 2)
        return;
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    int max_dims                           = tensor_shape_out->GetDimCount();
    if (pad_tensor_shape0.GetDimCount() < pad_tensor_shape1.GetDimCount()) {
        pad_tensor_shape0.SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape0->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape0.SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape0.SetDim(i, tensor_shape0->GetDim(i - offset));
        }
    } else {
        pad_tensor_shape1.SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape1->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape1.SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape1.SetDim(i, tensor_shape1->GetDim(i - offset));
        }
    }

    const int dimCount   = tensor_shape_out->GetDimCount();
    uint32_t stride0     = 1;
    uint32_t stride1     = 1;
    uint32_t stride_out0 = 1;

    for (int stride_pos = dimCount - 1; stride_pos >= 0; stride_pos--) {
        int i = stride_pos;
        if (stride_pos == dimCount - 1)
            i = 1;
        else if (stride_pos == 0)
            i = 0;
        else
            i = stride_pos + 1;
        stride_in0[stride_pos] = pad_tensor_shape0.GetDim(i) == 1 ? 0 : stride0;
        stride_in1[stride_pos] = pad_tensor_shape1.GetDim(i) == 1 ? 0 : stride1;
        stride_out[stride_pos] = stride_out0;
        if (i == 1) { // for channel dim, div packed_channel
            stride0 *= (pad_tensor_shape0.GetDim(i) + packed_channel - 1) / packed_channel;
            stride1 *= (pad_tensor_shape1.GetDim(i) + packed_channel - 1) / packed_channel;
            stride_out0 *= (tensor_shape_out->GetDim(i) + packed_channel - 1) / packed_channel;
        } else {
            stride0 *= pad_tensor_shape0.GetDim(i);
            stride1 *= pad_tensor_shape1.GetDim(i);
            stride_out0 *= tensor_shape_out->GetDim(i);
        }
    }
}

static void ppl_pad_tensor_shape(const ppl::nn::TensorShape* tensor_shape0,
                                 const ppl::nn::TensorShape* tensor_shape1,
                                 ppl::nn::TensorShape* pad_tensor_shape0,
                                 ppl::nn::TensorShape* pad_tensor_shape1)
{
    int max_dims = std::max(tensor_shape0->GetDimCount(), tensor_shape1->GetDimCount());
    if (pad_tensor_shape0->GetDimCount() < pad_tensor_shape1->GetDimCount()) {
        pad_tensor_shape0->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape0->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape0->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape0->SetDim(i, tensor_shape0->GetDim(i - offset));
        }
    } else {
        pad_tensor_shape1->SetDimCount(max_dims);
        // pad 1 to shape_min_pad's higher dim
        int offset = max_dims - tensor_shape1->GetDimCount();
        for (int i = 0; i < offset; i++) {
            pad_tensor_shape1->SetDim(i, 1);
        }
        for (int i = offset; i < max_dims; i++) {
            pad_tensor_shape1->SetDim(i, tensor_shape1->GetDim(i - offset));
        }
    }
}

static int ppl_get_num_broadcast_dims(const ppl::nn::TensorShape* tensor_shape0,
                                      const ppl::nn::TensorShape* tensor_shape1,
                                      int& aixs)
{
    ppl::nn::TensorShape pad_tensor_shape0 = *tensor_shape0;
    ppl::nn::TensorShape pad_tensor_shape1 = *tensor_shape1;
    ppl_pad_tensor_shape(tensor_shape0, tensor_shape1, &pad_tensor_shape0, &pad_tensor_shape1);
    int dim_count          = pad_tensor_shape0.GetDimCount();
    int num_broadcast_dims = 0;
    for (int it = 0; it < dim_count; ++it) {
        if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
            ++num_broadcast_dims;
    }
    if (num_broadcast_dims == 1) {
        for (int it = 0; it < dim_count; ++it) {
            if (pad_tensor_shape0.GetDim(it) != pad_tensor_shape1.GetDim(it))
                aixs = it;
        }
    }
    return num_broadcast_dims;
}

#define MAXDIMENSIONS 7

struct LogicalParam {
    uint32_t stride_in0[MAXDIMENSIONS];
    uint32_t stride_in1[MAXDIMENSIONS];
    uint32_t stride_out[MAXDIMENSIONS];
};

template <LogicalOpType op_type, typename T1, typename T2>
__global__ void ppl_cukernel_logical(
    const uint64_t num_elems,
    const int dim_count,
    LogicalParam param,
    const T1* input0,
    const T1* input1,
    T2* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    uint64_t out_index = index;
    uint64_t offset0   = 0;
    uint64_t offset1   = 0;
    for (int i = 0; i < dim_count; i++) {
        uint64_t dim_off = index / param.stride_out[i];
        offset0 += dim_off * param.stride_in0[i];
        offset1 += dim_off * param.stride_in1[i];
        index = index % param.stride_out[i];
    }
    output[out_index] = ppl_logical_vector<op_type>(input0[offset0], input1[offset1]);
#endif
}

template <LogicalOpType op_type, typename T1, typename T2>
__global__ void ppl_cukernel_logical_naive(
    const uint64_t num_elems,
    const T1* input0,
    const T1* input1,
    T2* output)
{
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    output[index] = ppl_logical_vector<op_type>(input0[index], input1[index]);
#endif
}

template <LogicalOpType op_type>
ppl::common::RetCode PPLCUDALogicalForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape0,
    const bool* input0,
    const ppl::nn::TensorShape* input_shape1,
    const bool* input1,
    const ppl::nn::TensorShape* output_shape,
    bool* output)
{
    LogicalParam param;

    uint64_t num_elems = output_shape->GetElementsIncludingPadding();
    int dim_count      = output_shape->GetDimCount();
    int block_size     = 256;

    int axis               = 0;
    int num_broadcast_dims = ppl_get_num_broadcast_dims(input_shape0, input_shape1, axis);
    switch (output_shape->GetDataFormat()) {
        case ppl::common::DATAFORMAT_NDARRAY: {
            if (num_broadcast_dims != 0) {
                int channel_shift  = 0;
                int packed_channel = 1;
                uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
                ppl_logical_prepare_strides(input_shape0, input_shape1, output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
                ppl_cukernel_logical<op_type, bool, bool><<<grid_size,
                                                            block_size,
                                                            0,
                                                            stream>>>(num_elems >> channel_shift, dim_count, param, (const bool*)input0, (const bool*)input1, (bool*)output);
            } else {
                uint64_t grid_size = (num_elems + block_size - 1) / block_size;
                ppl_cukernel_logical_naive<op_type, bool, bool><<<grid_size,
                                                                  block_size,
                                                                  0,
                                                                  stream>>>(num_elems, (const bool*)input0, (const bool*)input1, (bool*)output);
            }
            return ppl::common::RC_SUCCESS;
        }
        case ppl::common::DATAFORMAT_NHWC8: {
            bool can_broadcast = (input_shape0->GetDimCount() >= 2) && (input_shape1->GetDimCount() >= 2);
            if (!can_broadcast)
                return ppl::common::RC_UNSUPPORTED;
            if ((input_shape0->GetDim(1) & 0x7) || (input_shape1->GetDim(1) & 0x7))
                return ppl::common::RC_UNSUPPORTED;
            int channel_shift  = 3;
            int packed_channel = 8;
            uint64_t grid_size = ((num_elems >> channel_shift) + block_size - 1) / block_size;
            ppl_logical_prepare_strides_nhwc(input_shape0, input_shape1, output_shape, packed_channel, param.stride_in0, param.stride_in1, param.stride_out);
            ppl_cukernel_logical<op_type, bool8_, bool8_><<<grid_size, block_size, 0, stream>>>(num_elems >> channel_shift, dim_count, param, (const bool8_*)input0, (const bool8_*)input1, (bool8_*)output);
            return ppl::common::RC_SUCCESS;
        }
        default:
            return ppl::common::RC_UNSUPPORTED;
    }
#undef SWITCH_CASE
}

#define INSTANT(OPTYPE)                                                            \
    ppl::common::RetCode PPLCUDALogical##OPTYPE##ForwardImp(                       \
        cudaStream_t stream,                                                       \
        const ppl::nn::TensorShape* input_shape0,                                  \
        const bool* input0,                                                        \
        const ppl::nn::TensorShape* input_shape1,                                  \
        const bool* input1,                                                        \
        const ppl::nn::TensorShape* output_shape,                                  \
        bool* output)                                                              \
    {                                                                              \
        if (input_shape0->GetDataType() == ppl::common::DATATYPE_BOOL) {           \
            return PPLCUDALogicalForwardImp<Logical_##OPTYPE>(stream,              \
                                                              input_shape0,        \
                                                              (const bool*)input0, \
                                                              input_shape1,        \
                                                              (const bool*)input1, \
                                                              output_shape,        \
                                                              output);             \
        } else {                                                                   \
            return ppl::common::RC_UNSUPPORTED;                                    \
        }                                                                          \
    }

INSTANT(And);

#undef INSTANT
