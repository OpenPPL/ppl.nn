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

#include "cudakernel/nn/mmcv_nms.h"
#include "cudakernel/nn/topk.h"
#include "cudakernel/math/math.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/common/tensor_shape.h"
#include <cuda_fp16.h>
#include <float.h>
#include <memory>
#define CUDA_ALIGNMENT 128
constexpr int N_BOX_DIM      = 4;
constexpr int NMS_BLOCK_SIZE = 8 * sizeof(int64_t);

template <typename T>
__device__ __inline__ void maxMin(T a, T b, T &min_val, T &max_val)
{
    if (Math<T, T, T>::gt(a, b)) {
        min_val = b;
        max_val = a;
    } else {
        min_val = a;
        max_val = b;
    }
}

template <typename T>
__device__ __inline__ T Max(T a, T b)
{
    if (Math<T, T, T>::gt(a, b)) {
        return a;
    } else {
        return b;
    }
}

template <typename T>
__device__ __inline__ T Min(T a, T b)
{
    if (Math<T, T, T>::lt(a, b)) {
        return a;
    } else {
        return b;
    }
}

template <typename T>
__device__ bool devIoU(const T *box_i, const T *box_j, T iou_threshold, int offset)
{
    // use math helper
    typedef Math<T, T, T> OpMath;
    T ix_min, ix_max, iy_min, iy_max;
    T jx_min, jx_max, jy_min, jy_max;

    maxMin(box_i[0], box_i[2], iy_min, iy_max);
    maxMin(box_i[1], box_i[3], ix_min, ix_max);
    maxMin(box_j[0], box_j[2], jy_min, jy_max);
    maxMin(box_j[1], box_j[3], jx_min, jx_max);

    T interx_min, interx_max, intery_min, intery_max;
    interx_min = Max(ix_min, jx_min);
    intery_min = Max(iy_min, jy_min);
    interx_max = Min(ix_max, jx_max);
    intery_max = Min(iy_max, jy_max);

    T inter_area = OpMath::mul(Max(OpMath::add(OpMath::sub(interx_max, interx_min), (T)offset), (T)0),
                               Max(OpMath::add(OpMath::sub(intery_max, intery_min), (T)offset), (T)0));
    if (OpMath::le(inter_area, (T)0))
        return false;

    T i_area     = OpMath::mul(OpMath::add(OpMath::sub(ix_max, ix_min), (T)offset), OpMath::add(OpMath::sub(iy_max, iy_min), (T)offset));
    T j_area     = OpMath::mul(OpMath::add(OpMath::sub(jx_max, jx_min), (T)offset), OpMath::add(OpMath::sub(jy_max, jy_min), (T)offset));
    T union_area = OpMath::sub(OpMath::add(i_area, j_area), inter_area);
    if (OpMath::le(i_area, (T)0) || OpMath::le(j_area, (T)0) ||
        OpMath::le(union_area, (T)0))
        return false;

    T iou_ratio = OpMath::div(inter_area, union_area);
    return OpMath::gt(iou_ratio, iou_threshold);
}

template <typename T>
__global__ __launch_bounds__(NMS_BLOCK_SIZE) void mmcv_nms_one_one_kernel(
    int num_boxes,
    int num_blocks,
    const T *boxes,
    float iou_threshold,
    int offset,
    uint64_t *out_mask)
{
    T t_iou_threshold = (T)iou_threshold;
    __shared__ T s_boxes[NMS_BLOCK_SIZE * N_BOX_DIM];
    // step 1: load col boxes to shared memory
    int tid             = threadIdx.x;
    int col_boxes_start = blockIdx.x * NMS_BLOCK_SIZE;
    int row_boxes_start = blockIdx.y * NMS_BLOCK_SIZE;
    // no need to compute (redundant)
    if (col_boxes_start < row_boxes_start)
        return;
    // last thread block may overflow
    int col_size = min(num_boxes - col_boxes_start, NMS_BLOCK_SIZE);
    if (tid < col_size) {
        s_boxes[tid * N_BOX_DIM + 0] = boxes[(col_boxes_start + tid) * N_BOX_DIM + 0];
        s_boxes[tid * N_BOX_DIM + 1] = boxes[(col_boxes_start + tid) * N_BOX_DIM + 1];
        s_boxes[tid * N_BOX_DIM + 2] = boxes[(col_boxes_start + tid) * N_BOX_DIM + 2];
        s_boxes[tid * N_BOX_DIM + 3] = boxes[(col_boxes_start + tid) * N_BOX_DIM + 3];
    }
    __syncthreads();
    // step 2: iou mask with #NMS_BLOCK_SIZE boxes in smem
    int row_size = min(num_boxes - row_boxes_start, NMS_BLOCK_SIZE);
    if (tid < row_size) {
        uint64_t mask    = 0;
        int cur_box      = row_boxes_start + tid;
        int start        = (row_boxes_start == col_boxes_start) ? tid + 1 : 0;
        const T *box_row = boxes + cur_box * N_BOX_DIM;
        for (int it = start; it < col_size; ++it) {
            if (devIoU(box_row, s_boxes + it * N_BOX_DIM, t_iou_threshold, offset)) {
                mask |= (1ULL << it);
            }
        }
        int out_idx       = cur_box * num_blocks + blockIdx.x;
        out_mask[out_idx] = mask;
    }
}

__device__ __inline__ bool isBitSet(uint64_t *mask, int pos)
{
    constexpr int num_bits = 6; // int64_t
    constexpr int mod_num  = 63;
    int mask_pos           = pos >> num_bits; // div(64)
    int rem_pos            = pos & mod_num; // %(64)
    return (mask[mask_pos] >> rem_pos) & 1;
}

// only launch one thread block
__global__ void mmcv_nms_reduce_mask_kernel(
    int num_blocks,
    int num_boxes,
    const uint64_t *in_mask,
    bool *reduced_mask)
{
    extern __shared__ uint64_t s_reduced_mask[];
    int tid = threadIdx.x;
    for (int it = tid; it < num_blocks; it += blockDim.x) {
        s_reduced_mask[it] = 0xFFFFFFFFFFFFFFFF;
    }
    __syncthreads();
    // no need to deal with last box's mask: num_boxes - 1
    for (int b_idx = 0; b_idx < num_boxes - 1; ++b_idx) {
        if (!isBitSet(s_reduced_mask, b_idx))
            continue;
        const uint64_t *cur_mask = in_mask + b_idx * num_blocks;
        for (int it = tid; it < num_blocks; it += blockDim.x) {
            s_reduced_mask[it] &= ~cur_mask[it];
        }
        __syncthreads();
    }
    for (int it = tid; it < num_boxes; it += blockDim.x) {
        reduced_mask[it] = isBitSet(s_reduced_mask, it);
    }
}

__global__ void mmcv_nms_reduce_mask_kernel_global(
    int num_blocks,
    int num_boxes,
    const uint64_t *in_mask,
    uint64_t *g_reduce_mask,
    bool *reduced_mask)
{
    int tid = threadIdx.x;
    for (int it = tid; it < num_blocks; it += blockDim.x) {
        g_reduce_mask[it] = 0xFFFFFFFFFFFFFFFF;
    }
    __syncthreads();
    // no need to deal with last box's mask: num_boxes - 1
    for (int b_idx = 0; b_idx < num_boxes - 1; ++b_idx) {
        if (!isBitSet(g_reduce_mask, b_idx))
            continue;
        const uint64_t *cur_mask = in_mask + b_idx * num_blocks;
        for (int it = tid; it < num_blocks; it += blockDim.x) {
            g_reduce_mask[it] &= ~cur_mask[it];
        }
        __syncthreads();
    }
    for (int it = tid; it < num_boxes; it += blockDim.x) {
        reduced_mask[it] = isBitSet(g_reduce_mask, it);
    }
}

template <typename T>
__global__ void mmcv_indexSelectBoxes(
    int num_filtered_boxes,
    int32_t *sorted_indices,
    const T *boxes,
    T *sorted_boxes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_filtered_boxes)
        return;
    int in_index  = sorted_indices[index] * N_BOX_DIM;
    int out_index = index * N_BOX_DIM;
#pragma unrool
    for (int it = 0; it < N_BOX_DIM; ++it) {
        sorted_boxes[out_index + it] = boxes[in_index + it];
    }
}

template <typename T>
void MMCVNMSGpuImpl(
    cudaStream_t stream,
    const T *sorted_boxes,
    float iou_threshold,
    int num_boxes,
    int offset,
    int max_shared_mem,
    uint64_t *g_reduce_mask,
    uint64_t *dev_mask,
    bool *result_mask)
{
    // step 1: calculate all iou
    constexpr int block_size = NMS_BLOCK_SIZE;
    int num_blocks           = DivUp(num_boxes, block_size);
    dim3 grid_size(num_blocks, num_blocks);
    mmcv_nms_one_one_kernel<<<grid_size, block_size, 0, stream>>>(num_boxes,
                                                                  num_blocks,
                                                                  sorted_boxes,
                                                                  iou_threshold,
                                                                  offset,
                                                                  dev_mask);
    // step 2: mask reduce
    int32_t reduced_mask_size = num_blocks * block_size;
    if (max_shared_mem > reduced_mask_size) {
        // #boxes should not exceed #bits in shared memory
        mmcv_nms_reduce_mask_kernel<<<1, 1024, reduced_mask_size, stream>>>(num_blocks,
                                                                            num_boxes,
                                                                            dev_mask,
                                                                            result_mask);
    } else {
        // use global memory
        mmcv_nms_reduce_mask_kernel_global<<<1, 1024, 0, stream>>>(num_blocks,
                                                                   num_boxes,
                                                                   dev_mask,
                                                                   g_reduce_mask,
                                                                   result_mask);
    }
}

int64_t PPLMMCVNMSGetTempBufferSize(const ppl::nn::TensorShape *boxes_shape)
{
    int64_t total_size = 0;
    int elem_size      = ppl::common::GetSizeOfDataType(boxes_shape->GetDataType());
    int num_boxes      = boxes_shape->GetDim(0);
    ppl::nn::TensorShape indices_shape;
    indices_shape.SetDataType(ppl::common::DATATYPE_INT64); // max int64
    indices_shape.Reshape({num_boxes});
    int axis = 0;
    total_size += Align(PPLTopKGetTempBufferSize(&indices_shape, num_boxes, axis), CUDA_ALIGNMENT);

    total_size += Align(elem_size * num_boxes, CUDA_ALIGNMENT); // sorted scores;
    total_size += Align(sizeof(int64_t) * num_boxes, CUDA_ALIGNMENT); // sorted indices;
    total_size += Align(elem_size * N_BOX_DIM * num_boxes, CUDA_ALIGNMENT); // sorted boxes;
    int blocks = DivUp(num_boxes, NMS_BLOCK_SIZE);
    total_size += Align(sizeof(uint64_t) * blocks * num_boxes, CUDA_ALIGNMENT); // one-one mapping mask;
    total_size += Align(sizeof(bool) * num_boxes, CUDA_ALIGNMENT); // reduced mask;

    // reduce needed
    int num_blocks = DivUp(num_boxes, NMS_BLOCK_SIZE);
    total_size += Align(num_blocks * NMS_BLOCK_SIZE, CUDA_ALIGNMENT); // count filtered boxes number

    return total_size;
}

ppl::common::RetCode PPLCUDAMMCVNMSForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape *boxes_shape,
    const void *boxes,
    ppl::nn::TensorShape *scores_shape,
    const void *scores,
    ppl::nn::TensorShape *output_shape,
    int64_t *output,
    void *temp_buffer,
    int64_t temp_buffer_bytes,
    int device_id,
    float iou_threshold,
    int64_t offset)
{
    int num_selected_indices = 0;
    int num_boxes            = boxes_shape->GetDim(0);
    // shape for top-k use
    int elem_size            = ppl::common::GetSizeOfDataType(scores_shape->GetDataType());
    cudaDeviceProp gpu_prob;
    cudaGetDeviceProperties(&gpu_prob, device_id);
    int max_shared_mem = gpu_prob.sharedMemPerBlock;

    // temp buffer for sort & nms & construct
    ppl::nn::TensorShape topk_shape;
    topk_shape.SetDataType(scores_shape->GetDataType());
    topk_shape.Reshape({num_boxes});
    ppl::nn::TensorShape indices_shape;
    indices_shape.SetDataType(ppl::common::DATATYPE_INT32);
    indices_shape.Reshape({num_boxes});
    int axis                = 0;
    int topk_buffer_size    = PPLTopKGetTempBufferSize(&indices_shape, num_boxes, axis);
    void *topk_buffer       = temp_buffer;
    void *sorted_scores     = static_cast<void *>(static_cast<char *>(topk_buffer) + Align(topk_buffer_size, CUDA_ALIGNMENT));
    int32_t *sorted_indices = reinterpret_cast<int32_t *>(
        static_cast<char *>(sorted_scores) + Align(elem_size * num_boxes, CUDA_ALIGNMENT));
    void *sorted_boxes = static_cast<void *>((char *)sorted_indices + Align(num_boxes * sizeof(int32_t), CUDA_ALIGNMENT));
    uint64_t *dev_mask = reinterpret_cast<uint64_t *>(
        static_cast<char *>(sorted_boxes) + Align(elem_size * N_BOX_DIM * num_boxes, CUDA_ALIGNMENT));
    // each bit in int64_t represent one iou
    int blocks              = DivUp(num_boxes, NMS_BLOCK_SIZE);
    bool *result_mask       = reinterpret_cast<bool *>((char *)dev_mask + Align(blocks * num_boxes * sizeof(uint64_t), CUDA_ALIGNMENT));
    uint64_t *g_reduce_mask = reinterpret_cast<uint64_t *>((char *)result_mask + Align(sizeof(bool) * num_boxes, CUDA_ALIGNMENT));

    // reset to zero each iteration
    cudaMemset(temp_buffer, 0, temp_buffer_bytes);

    // step 1: sort scores and index select boxes
    PPLCUDATopKForwardImp(stream,
                          &topk_shape,
                          static_cast<const float *>(scores),
                          &topk_shape,
                          sorted_scores,
                          &indices_shape,
                          sorted_indices,
                          topk_buffer,
                          topk_buffer_size,
                          num_boxes,
                          axis);
    // index select
    {
        int block_size = 256;
        int grid_size  = DivUp(num_boxes, block_size);
        if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
            mmcv_indexSelectBoxes<<<grid_size, block_size, 0, stream>>>(
                num_boxes, sorted_indices, static_cast<const float *>(boxes), (float *)sorted_boxes);
        } else if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
            mmcv_indexSelectBoxes<<<grid_size, block_size, 0, stream>>>(
                num_boxes, sorted_indices, static_cast<const half *>(boxes), (half *)sorted_boxes);
        } else {
            return ppl::common::RC_UNSUPPORTED;
        }
    }
    // step 2: nms operations (type related)
    if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        MMCVNMSGpuImpl<float>(stream, (const float *)sorted_boxes, iou_threshold, num_boxes, offset, max_shared_mem, g_reduce_mask, dev_mask, result_mask);
    } else if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        MMCVNMSGpuImpl<half>(stream, (const half *)sorted_boxes, iou_threshold, num_boxes, offset, max_shared_mem, g_reduce_mask, dev_mask, result_mask);
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    // step 3: mapping back to origin index on cpu
    std::unique_ptr<int32_t[]> h_sorted_indices(new int32_t[num_boxes]);
    std::unique_ptr<bool[]> h_result_mask(new bool[num_boxes]);
    // construct output on cpu
    std::unique_ptr<int64_t[]> h_constructed_indices(new int64_t[num_boxes]);
    cudaMemcpyAsync(h_sorted_indices.get(), sorted_indices, sizeof(int32_t) * num_boxes, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_result_mask.get(), result_mask, sizeof(bool) * num_boxes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int num_eval_boxes = num_boxes;
    for (int it = 0; it < num_eval_boxes; ++it) {
        if (h_result_mask.get()[it]) {
            h_constructed_indices.get()[num_selected_indices] =
                h_sorted_indices.get()[it];
            ++num_selected_indices;
        }
    }
    // step 4: gather one class output to totals
    cudaMemcpyAsync(output, h_constructed_indices.get(),
                    // 3 means [batch_index, class_index, box_index]
                    sizeof(int64_t) * num_selected_indices,
                    cudaMemcpyHostToDevice,
                    stream);

    output_shape->SetDim(0, num_selected_indices);
    return ppl::common::RC_SUCCESS;
}
