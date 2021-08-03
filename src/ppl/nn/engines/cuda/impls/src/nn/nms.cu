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

#include "cudakernel/nn/nms.h"
#include "cudakernel/nn/topk.h"
#include "cudakernel/math/math.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/common/tensor_shape.h"
#include <cuda_fp16.h>
#include <float.h>
#include <memory>
#define CUDA_ALIGNMENT    128
constexpr int N_BOX_DIM      = 4;
constexpr int NMS_BLOCK_SIZE = 8 * sizeof(int64_t);

template <typename T>
__device__ __inline__ void maxMin(T a, T b, T& min_val, T& max_val)
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
__device__ bool devIoU(const T* box_i, const T* box_j, T iou_threshold, int center_point_box)
{
    // use math helper
    typedef Math<T, T, T> OpMath;
    T ix_min, ix_max, iy_min, iy_max;
    T jx_min, jx_max, jy_min, jy_max;

    if (center_point_box == 0) {
        maxMin(box_i[0], box_i[2], iy_min, iy_max);
        maxMin(box_i[1], box_i[3], ix_min, ix_max);
        maxMin(box_j[0], box_j[2], jy_min, jy_max);
        maxMin(box_j[1], box_j[3], jx_min, jx_max);
    } else {
        T iw_half = OpMath::div(box_i[2], (T)2);
        T ih_half = OpMath::div(box_i[3], (T)2);
        ix_min    = OpMath::sub(box_i[0], iw_half);
        ix_max    = OpMath::add(box_i[0], iw_half);
        iy_min    = OpMath::sub(box_i[1], ih_half);
        iy_max    = OpMath::add(box_i[1], ih_half);

        T jw_half = OpMath::div(box_j[2], (T)2);
        T jh_half = OpMath::div(box_j[3], (T)2);
        jx_min    = OpMath::sub(box_j[0], jw_half);
        jx_max    = OpMath::add(box_j[0], jw_half);
        jy_min    = OpMath::sub(box_j[1], jh_half);
        jy_max    = OpMath::add(box_j[1], jh_half);
    }

    T interx_min, interx_max, intery_min, intery_max;
    interx_min = Max(ix_min, jx_min);
    intery_min = Max(iy_min, jy_min);
    interx_max = Min(ix_max, jx_max);
    intery_max = Min(iy_max, jy_max);

    T inter_area = OpMath::mul(Max(OpMath::sub(interx_max, interx_min), (T)0),
                               Max(OpMath::sub(intery_max, intery_min), (T)0));
    if (OpMath::le(inter_area, (T)0))
        return false;

    T i_area     = OpMath::mul(OpMath::sub(ix_max, ix_min), OpMath::sub(iy_max, iy_min));
    T j_area     = OpMath::mul(OpMath::sub(jx_max, jx_min), OpMath::sub(jy_max, jy_min));
    T union_area = OpMath::sub(OpMath::add(i_area, j_area), inter_area);
    if (OpMath::le(i_area, (T)0) || OpMath::le(j_area, (T)0) ||
        OpMath::le(union_area, (T)0))
        return false;

    T iou_ratio = OpMath::div(inter_area, union_area);
    return OpMath::gt(iou_ratio, iou_threshold);
}

template <typename T>
__global__ __launch_bounds__(NMS_BLOCK_SIZE) void nms_one_one_kernel(
    int num_boxes,
    int num_blocks,
    const T* boxes,
    float iou_threshold,
    int center_point_box,
    uint64_t* out_mask)
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
            if (devIoU(box_row, s_boxes + it * N_BOX_DIM, t_iou_threshold, center_point_box)) {
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
__global__ void nms_reduce_mask_kernel(
    int num_blocks,
    int num_boxes,
    int max_boxes,
    const uint64_t* in_mask,
    bool* reduced_mask)
{
    extern __shared__ uint64_t s_reduced_mask[];
    int tid = threadIdx.x;
    for (int it = tid; it < num_blocks; it += blockDim.x) {
        s_reduced_mask[it] = 0xFFFFFFFFFFFFFFFF;
    }
    __syncthreads();
    int accepted_boxes = 0;
    // no need to deal with last box's mask: num_boxes - 1
    for (int b_idx = 0; b_idx < num_boxes - 1; ++b_idx) {
        if (!isBitSet(s_reduced_mask, b_idx))
            continue;
        ++accepted_boxes;
        const uint64_t *cur_mask = in_mask + b_idx * num_blocks;
        for (int it = tid; it < num_blocks; it += blockDim.x) {
            s_reduced_mask[it] &= ~cur_mask[it];
        }
        __syncthreads();
        if (accepted_boxes >= max_boxes)
            break;
    }
    for (int it = tid; it < num_boxes; it += blockDim.x) {
        reduced_mask[it] = isBitSet(s_reduced_mask, it);
    }
}

__global__ void nms_reduce_mask_kernel_global(
    int num_blocks,
    int num_boxes,
    int max_boxes,
    const uint64_t* in_mask,
    uint64_t* g_reduce_mask,
    bool* reduced_mask)
{
    int tid = threadIdx.x;
    for (int it = tid; it < num_blocks; it += blockDim.x) {
        g_reduce_mask[it] = 0xFFFFFFFFFFFFFFFF;
    }
    __syncthreads();
    int accepted_boxes = 0;
    // no need to deal with last box's mask: num_boxes - 1
    for (int b_idx = 0; b_idx < num_boxes - 1; ++b_idx) {
        if (!isBitSet(g_reduce_mask, b_idx))
            continue;
        ++accepted_boxes;
        const uint64_t *cur_mask = in_mask + b_idx * num_blocks;
        for (int it = tid; it < num_blocks; it += blockDim.x) {
            g_reduce_mask[it] &= ~cur_mask[it];
        }
        __syncthreads();
        if (accepted_boxes >= max_boxes)
            break;
    }
    for (int it = tid; it < num_boxes; it += blockDim.x) {
        reduced_mask[it] = isBitSet(g_reduce_mask, it);
    }
}

template <typename T>
__global__ void indexSelectBoxes(
    int num_filtered_boxes,
    int32_t* sorted_indices,
    const T* boxes,
    T* sorted_boxes)
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

// only launch 1 thread blocks (256 threads)
template <typename T>
__global__ void countScoresBoxes(
    int num_boxes,
    float score_threshold,
    const T* sorted_boxes,
    int* d_num_filter_scores)
{
    int tid             = threadIdx.x;
    T t_score_threshold = (T)score_threshold;
    __shared__ int acc_num[256];
    acc_num[tid] = 0;
    __syncthreads();
    for (int it = tid; it < num_boxes; it += blockDim.x) {
        acc_num[tid] += Math<T, T, T>::gt(sorted_boxes[it], t_score_threshold) ? 1 : 0;
    }
    for (int it = 128; it > 0; it = it >> 1) {
        if (tid < it)
            acc_num[tid] += acc_num[tid + it];
        __syncthreads();
    }

    if (tid == 0)
        d_num_filter_scores[0] = acc_num[0];
}

template <typename T>
void NMSGpuImpl(
    cudaStream_t stream,
    const T* sorted_boxes,
    float iou_threshold,
    int num_filtered_boxes,
    int max_output_boxes_per_class,
    int center_point_box,
    int max_shared_mem,
    uint64_t* g_reduce_mask,
    uint64_t* dev_mask,
    bool* result_mask)
{
    // step 1: calculate all iou
    constexpr int block_size = NMS_BLOCK_SIZE;
    int num_blocks           = DivUp(num_filtered_boxes, block_size);
    dim3 grid_size(num_blocks, num_blocks);
    nms_one_one_kernel<<<grid_size, block_size, 0, stream>>>(num_filtered_boxes,
                                                             num_blocks,
                                                             sorted_boxes,
                                                             iou_threshold,
                                                             center_point_box,
                                                             dev_mask);
    // step 2: mask reduce
    int32_t reduced_mask_size = num_blocks * block_size;
    if (max_shared_mem > reduced_mask_size) {
    // #boxes should not exceed #bits in shared memory
        nms_reduce_mask_kernel<<<1, 1024, reduced_mask_size, stream>>>(num_blocks,
                                                                    num_filtered_boxes,
                                                                    max_output_boxes_per_class,
                                                                    dev_mask,
                                                                    result_mask);
    } else {
        // use global memory
        nms_reduce_mask_kernel_global<<<1, 1024, 0, stream>>>(num_blocks,
                                                                num_filtered_boxes,
                                                                max_output_boxes_per_class,
                                                                dev_mask,
                                                                g_reduce_mask,
                                                                result_mask);
    }
}

int64_t PPLNMSGetTempBufferSize(const ppl::nn::TensorShape* scores_shape)
{
    int64_t total_size = 0;
    int elem_size      = ppl::common::GetSizeOfDataType(scores_shape->GetDataType());
    int num_class      = scores_shape->GetDim(1);
    int num_boxes      = scores_shape->GetDim(2);
    ppl::nn::TensorShape indices_shape(*scores_shape);
    indices_shape.Reshape({num_class, num_boxes});
    indices_shape.SetDataType(ppl::common::DATATYPE_INT64); // max int64
    int axis = 1;
    total_size += Align(PPLTopKGetTempBufferSize(&indices_shape, num_boxes, axis), CUDA_ALIGNMENT);
    total_size += Align(elem_size * num_class * num_boxes, CUDA_ALIGNMENT); // sorted scores;
    total_size += Align(sizeof(int64_t) * num_class * num_boxes, CUDA_ALIGNMENT); // sorted indices;

    total_size += Align(elem_size * N_BOX_DIM * num_boxes, CUDA_ALIGNMENT); // sorted boxes;
    int blocks = DivUp(num_boxes, NMS_BLOCK_SIZE);
    total_size += Align(sizeof(uint64_t) * blocks * num_boxes, CUDA_ALIGNMENT); // one-one mapping mask;
    total_size += Align(sizeof(bool) * num_boxes, CUDA_ALIGNMENT); // reduced mask;
    total_size += Align(sizeof(int), CUDA_ALIGNMENT); // count filtered boxes number

    // reduce needed
    int num_blocks           = DivUp(num_boxes, NMS_BLOCK_SIZE);
    total_size += Align(num_blocks * NMS_BLOCK_SIZE, CUDA_ALIGNMENT); // count filtered boxes number

    return total_size;
}

ppl::common::RetCode PPLCUDANMSForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* boxes_shape,
    const void* boxes,
    ppl::nn::TensorShape* scores_shape,
    const void* scores,
    ppl::nn::TensorShape* output_shape,
    int64_t* output,
    void* temp_buffer,
    int64_t temp_buffer_bytes,
    int device_id,
    int center_point_box,
    int max_output_boxes_per_class,
    float iou_threshold,
    float score_threshold)
{
    int num_selected_indices = 0;
    int num_batch            = boxes_shape->GetDim(0);
    int num_boxes            = boxes_shape->GetDim(1);
    int num_class            = scores_shape->GetDim(1);
    // shape for top-k use
    int elem_size            = ppl::common::GetSizeOfDataType(scores_shape->GetDataType());
    cudaDeviceProp gpu_prob;
    cudaGetDeviceProperties(&gpu_prob, device_id);
    int max_shared_mem = gpu_prob.sharedMemPerBlock;

    // temp buffer for sort & nms & construct
    ppl::nn::TensorShape topk_shape(*scores_shape);
    topk_shape.Reshape({num_class, num_boxes});
    topk_shape.SetDataType(scores_shape->GetDataType());
    ppl::nn::TensorShape indices_shape(*scores_shape);
    indices_shape.Reshape({num_class, num_boxes});
    indices_shape.SetDataType(ppl::common::DATATYPE_INT32);
    int axis = 1;
    int topk_buffer_size    = PPLTopKGetTempBufferSize(&indices_shape, num_boxes, axis);
    void *topk_buffer       = temp_buffer;
    void *sorted_scores_tot     = static_cast<void *>(static_cast<char *>(topk_buffer) + Align(topk_buffer_size, CUDA_ALIGNMENT));
    int32_t *sorted_indices_tot = reinterpret_cast<int32_t *>(
        static_cast<char *>(sorted_scores_tot) + Align(elem_size * num_class * num_boxes, CUDA_ALIGNMENT));

    void *sorted_boxes = static_cast<void *>((char *)sorted_indices_tot + Align(num_class * num_boxes * sizeof(int32_t), CUDA_ALIGNMENT));
    uint64_t *dev_mask = reinterpret_cast<uint64_t *>(
        static_cast<char *>(sorted_boxes) + Align(elem_size * N_BOX_DIM * num_boxes, CUDA_ALIGNMENT));
    // each bit in int64_t represent one iou
    int blocks               = DivUp(num_boxes, NMS_BLOCK_SIZE);
    bool *result_mask        = reinterpret_cast<bool *>((char *)dev_mask + Align(blocks * num_boxes * sizeof(uint64_t), CUDA_ALIGNMENT));
    int *d_num_filter_scores = reinterpret_cast<int *>(result_mask + Align(num_boxes * sizeof(bool), CUDA_ALIGNMENT));
    uint64_t *g_reduce_mask  = reinterpret_cast<uint64_t *>((char *)d_num_filter_scores + Align(sizeof(int), CUDA_ALIGNMENT));
    int dev_mask_size = Align(blocks * num_boxes * sizeof(uint64_t), CUDA_ALIGNMENT);
    // process one class one time
    for (int b = 0; b < num_batch; ++b) {
        // step 1: sort scores and index select boxes
        cudaMemset(temp_buffer, 0, temp_buffer_bytes);
        PPLCUDATopKForwardImp(stream,
                            &topk_shape,
                            static_cast<const float *>(scores) + b * scores_shape->GetElementsFromDimensionIncludingPadding(1),
                            &topk_shape,
                            sorted_scores_tot,
                            &indices_shape,
                            sorted_indices_tot,
                            topk_buffer,
                            topk_buffer_size,
                            num_boxes,
                            axis);
        // int nms_buffer_size = temp_buffer_bytes - Align(topk_buffer_size, CUDA_ALIGNMENT) -
                // Align(elem_size * num_class * num_boxes, CUDA_ALIGNMENT) -
                // Align(num_class * num_boxes * sizeof(int32_t), CUDA_ALIGNMENT);

        for (int c = 0; c < num_class; ++c) {
            // reset to zero each iteration (Not necessary)
            // cudaMemset(sorted_boxes, 0, nms_buffer_size);
            float *sorted_scores = static_cast<float*>(sorted_scores_tot) + c * num_boxes;//+ b * num_class * num_boxes + c * num_boxes;
            int32_t *sorted_indices = sorted_indices_tot + c * num_boxes;// + b * num_class * num_boxes + c * num_boxes;

            int num_selected_indices_per_class = 0;
            int num_filtered_boxes             = 0;

            // count scores above score_threshold
            {
                int block_size = 256;
                if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
                    countScoresBoxes<<<1, block_size, 0, stream>>>(
                        num_boxes, score_threshold, (const float *)sorted_scores, d_num_filter_scores);
                } else if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                    countScoresBoxes<<<1, block_size, 0, stream>>>(
                        num_boxes, score_threshold, (const half *)sorted_scores, d_num_filter_scores);
                } else {
                    return ppl::common::RC_UNSUPPORTED;
                }
                cudaMemcpyAsync((void *)(&num_filtered_boxes), d_num_filter_scores, sizeof(int), cudaMemcpyDeviceToHost, stream);
            }
            cudaStreamSynchronize(stream);
            // index select
            if (num_filtered_boxes > 0) {
                {
                    int block_size = 256;
                    int grid_size  = DivUp(num_filtered_boxes, block_size);
                    if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
                        indexSelectBoxes<<<grid_size, block_size, 0, stream>>>(
                            num_filtered_boxes, sorted_indices, static_cast<const float *>(boxes) + b * num_boxes * 4, (float *)sorted_boxes);
                    } else if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                        indexSelectBoxes<<<grid_size, block_size, 0, stream>>>(
                            num_filtered_boxes, sorted_indices, static_cast<const half *>(boxes) + b * num_boxes * 4, (half *)sorted_boxes);
                    } else {
                        return ppl::common::RC_UNSUPPORTED;
                    }
                }
                // step 2: nms operations (type related)
                cudaMemset(dev_mask, 0, dev_mask_size);
                if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
                    NMSGpuImpl<float>(stream, (const float *)sorted_boxes, iou_threshold, num_filtered_boxes, max_output_boxes_per_class, center_point_box, max_shared_mem, g_reduce_mask, dev_mask, result_mask);
                } else if (boxes_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
                    NMSGpuImpl<half>(stream, (const half *)sorted_boxes, iou_threshold, num_filtered_boxes, max_output_boxes_per_class, center_point_box, max_shared_mem, g_reduce_mask, dev_mask, result_mask);
                } else {
                    return ppl::common::RC_UNSUPPORTED;
                }
                // step 3: mapping back to origin index on cpu
                std::unique_ptr<int32_t[]> h_sorted_indices(new int32_t[num_boxes]);
                std::unique_ptr<bool[]> h_result_mask(new bool[num_boxes]);
                // construct output on cpu
                std::unique_ptr<int64_t[]> h_constructed_indices(new int64_t[num_boxes * 3]);
                cudaMemcpyAsync(h_sorted_indices.get(), sorted_indices, sizeof(int32_t) * num_boxes, cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(h_result_mask.get(), result_mask, sizeof(bool) * num_boxes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                // int num_eval_boxes = std::min(num_filtered_boxes, max_output_boxes_per_class);
                for (int it = 0; it < num_filtered_boxes; ++it) {
                    if (h_result_mask.get()[it]) {
                        h_constructed_indices.get()[num_selected_indices_per_class * 3 + 0] = b;
                        h_constructed_indices.get()[num_selected_indices_per_class * 3 + 1] = c;
                        h_constructed_indices.get()[num_selected_indices_per_class * 3 + 2] =
                            h_sorted_indices.get()[it];
                        ++num_selected_indices_per_class;
                        if (num_selected_indices_per_class >= max_output_boxes_per_class) break;
                    }
                }
                // step 4: gather one class output to totals
                cudaMemcpyAsync(output + num_selected_indices * 3, h_constructed_indices.get(),
                                // 3 means [batch_index, class_index, box_index]
                                sizeof(int64_t) * num_selected_indices_per_class * 3,
                                cudaMemcpyHostToDevice,
                                stream);
                num_selected_indices += num_selected_indices_per_class;
            }
        }
    }
    output_shape->SetDim(0, num_selected_indices);
    return ppl::common::RC_SUCCESS;
}
