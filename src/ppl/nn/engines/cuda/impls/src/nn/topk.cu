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

#include "cudakernel/nn/topk.h"
#include "cudakernel/math/math.h"
#include "cudakernel/common/common.h"
#include "ppl/common/types.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>
#include <float.h>
#include <memory>

#define MAX_DIM           8 // tensor_shape.kMaxNumDimensions
#define killWARDependency 1

#define RADIX_SIZE 16
#define RADIX_BITS 4
const int RADIX_MASK = RADIX_SIZE - 1;
#define SORT_RADIX_SIZE 4
#define SORT_RADIX_BITS 2

struct TensorInfo {
    int shape[MAX_DIM];
    int strides[MAX_DIM];
    const void *data;
    int dims;
    TensorInfo(ppl::nn::TensorShape *tensor_shape, const void *data_ptr)
    {
        for (unsigned int i = 0; i < tensor_shape->GetDimCount() && i < MAX_DIM; i++) {
            shape[i] = tensor_shape->GetDim(i);
        }
        for (unsigned int i = tensor_shape->GetDimCount(); i < MAX_DIM; i++) {
            shape[i] = 1;
        }
        dims = tensor_shape->GetDimCount();
        data = data_ptr;
    }
};

__device__ __inline__ int convert_f2u(float v)
{
    unsigned int x    = __float_as_uint(v);
    unsigned int mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    unsigned int res  = x ^ mask;
    return res;
}
__device__ __inline__ float convert_u2f(int v)
{
    unsigned int mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    unsigned int x    = v ^ mask;
    return __uint_as_float(x);
}

template <typename T>
__device__ unsigned int convert2u(T value);

template <>
__device__ unsigned int convert2u(__half value)
{
    // must use short, for reverse convert
    unsigned short int x    = __half_as_ushort(value);
    unsigned short int mask = (x & 0x8000) ? 0xffff : 0x8000;
    unsigned int res        = x ^ mask;
    return res;
}
template <>
__device__ unsigned int convert2u(float value)
{
    unsigned int x    = __float_as_uint(value);
    unsigned int mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    unsigned int res  = x ^ mask;
    return res;
}
template <typename T>
__device__ T convertu2(unsigned int value);

template <>
__device__ __half convertu2(unsigned int value)
{
    unsigned short int sht  = (unsigned short int)value;
    unsigned short int mask = (sht & 0x8000) ? 0x8000 : 0xffff;
    unsigned short int x    = sht ^ mask;
    return __ushort_as_half(x);
}
template <>
__device__ float convertu2(unsigned int value)
{
    unsigned int mask = (value & 0x80000000) ? 0x80000000 : 0xffffffff;
    unsigned int x    = value ^ mask;
    return __uint_as_float(x);
}

// shape:[n,c,h,w]
__device__ int get_offset(int linearIdx, int Dims, TensorInfo info)
{
    int offset = 0;
    for (int i = Dims - 1; i > 0; --i) {
        int curDimIdx = linearIdx % info.shape[i];
        int curOffset = curDimIdx * info.strides[i];
        linearIdx /= info.shape[i];
        offset += curOffset;
    }
    return offset + linearIdx * info.strides[0];
}

template <typename T>
__device__ unsigned int find_desired(
    unsigned int *smem,
    int lane,
    const unsigned int mask,
    const unsigned int desired,
    const int inputSliceStride,
    const T *inputSlice,
    const int sliceSize)
{
    if (threadIdx.x == 0) {
        smem[0] = 0;
    }
    __syncthreads();

    for (int off = threadIdx.x; off - lane < sliceSize; off += blockDim.x) {
        bool inRange          = off < sliceSize;
        T value               = inRange ? inputSlice[off * inputSliceStride] : (T)0;
        unsigned int intValue = convert2u<T>(value);
        bool flag             = inRange && ((intValue & mask) == desired);
        if (flag) {
            smem[0] = 1;
            smem[1] = intValue;
        }
        __syncthreads();

        unsigned int isFound = smem[0];
        intValue             = smem[1];

        if (isFound) {
            return intValue;
        }
    }
    return 0;
}

template <typename T, bool dir>
__device__ T find_kth_value(
    int *smem,
    int K,
    const int sliceSize,
    const T *inputSlice,
    const int inputSliceStride)
{
    int count[RADIX_SIZE];
    // use fixed higher bits to filter data
    unsigned int mask    = 0; // fixed high bit
    unsigned int desired = 0; // current radix bits to fix
    int *radix_hist      = smem;
    unsigned int kthValue;
    for (int pos = 8 * sizeof(int) - RADIX_BITS; pos >= 0; pos -= RADIX_BITS) {
        // reinit radix_hist to 0 every loop
        for (int i = 0; i < RADIX_SIZE; i++) {
            count[i] = 0;
        }
        if (threadIdx.x < RADIX_SIZE) {
            radix_hist[threadIdx.x] = 0;
        }
        __syncthreads();

        const int lane = threadIdx.x & 31;
        for (int off = threadIdx.x; off - lane < sliceSize; off += blockDim.x) {
            bool inRange          = off < sliceSize;
            T value               = inRange ? inputSlice[off * inputSliceStride] : (T)0;
            unsigned int active   = __ballot_sync(0xffffffff, inRange);
            unsigned int intValue = convert2u<T>(value);

            // filter with desired
            bool inRadix = inRange && ((intValue & mask) == desired);
            int valueRadix;
            asm("bfe.u32 %0, %1, %2, %3;"
                : "=r"(valueRadix)
                : "r"(intValue), "r"(pos), "r"(RADIX_BITS));
#pragma unroll
            for (int i = 0; i < RADIX_SIZE; i++) {
                bool flag           = inRadix && (valueRadix == i);
                unsigned int ballot = __ballot_sync(active, flag);
                count[i] += __popc(ballot);
            }
        }
        if ((threadIdx.x & 31) == 0) {
            for (int i = 0; i < RADIX_SIZE; i++) {
                atomicAdd(radix_hist + i, count[i]);
            }
        }
        __syncthreads();

        // all threads in blk are the same
        for (int i = 0; i < RADIX_SIZE; i++) {
            count[i] = radix_hist[i];
        }
        if (killWARDependency) {
            __syncthreads();
        }

        // search K count
        if (dir == 1) { // topK largest
            for (int i = RADIX_SIZE - 1; i >= 0; --i) {
                if (K == count[i] && K == 1) {
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "+r"(desired)
                        : "r"(i), "r"(pos), "r"(RADIX_BITS));
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "+r"(mask)
                        : "r"(RADIX_MASK), "r"(pos), "r"(RADIX_BITS));
                    kthValue      = find_desired<T>((unsigned int *)smem, threadIdx.x, mask, desired, inputSliceStride, inputSlice, sliceSize);
                    T fp_kthValue = convertu2<T>(kthValue);
                    return fp_kthValue;
                } else if (K <= count[i]) { // narrow radix unitl K == count[i] == 1
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "=r"(desired)
                        : "r"(i), "r"(pos), "r"(RADIX_BITS));
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "=r"(mask)
                        : "r"(RADIX_MASK), "r"(pos), "r"(RADIX_BITS));
                    break;
                }

                K -= count[i];
            }
        } else {
            for (int i = 0; i < RADIX_SIZE; ++i) {
                if (K == count[i] && K == 1) {
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "+r"(desired)
                        : "r"(i), "r"(pos), "r"(RADIX_BITS));
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "+r"(mask)
                        : "r"(RADIX_MASK), "r"(pos), "r"(RADIX_BITS));
                    kthValue      = find_desired<T>((unsigned int *)smem, threadIdx.x, mask, desired, inputSliceStride, inputSlice, sliceSize);
                    T fp_kthValue = convertu2<T>(kthValue);
                    return fp_kthValue;
                } else if (K <= count[i]) { // narrow radix unitl K == count[i] == 1
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "+r"(desired)
                        : "r"(i), "r"(pos), "r"(RADIX_BITS));
                    asm("bfi.b32 %0, %1, %0, %2, %3;"
                        : "+r"(mask)
                        : "r"(RADIX_MASK), "r"(pos), "r"(RADIX_BITS));
                    break;
                }

                K -= count[i];
            }
        }
    }
    kthValue      = desired;
    T fp_kthValue = convertu2<T>(kthValue);
    return fp_kthValue;
}

template <typename T>
__device__ T scanInWarp(T value, int lane);
__device__ void prefix_scan(
    int *smem,
    const unsigned int active,
    const int activeWarps,
    const bool flag,
    int &index,
    int &blkTotal)
{
    if (threadIdx.x < blockDim.x / 32) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    unsigned int ballot   = __ballot_sync(active, flag);
    int lane              = threadIdx.x & 31;
    unsigned int laneMask = ~(0xffffffff << lane);
    laneMask              = active & laneMask;
    int warpId            = threadIdx.x >> 5;
    unsigned int leader   = __ffs(active) - 1;
    int total             = __popc(ballot);
    int prefix            = __popc(laneMask & ballot);

    if (lane == leader) {
        smem[warpId] = total;
    }
    __syncthreads();

    int warpOff = 0;
    if (threadIdx.x < 32) {
        int value         = smem[threadIdx.x];
        int warpPrefix    = scanInWarp<int>(value, lane);
        smem[threadIdx.x] = warpPrefix;
    }
    __syncthreads();

    if (warpId >= 1)
        warpOff = smem[warpId - 1];
    blkTotal = smem[activeWarps - 1];

    if (flag) {
        index = warpOff + prefix;
    }
    // write-after-read dependency
    if (killWARDependency) {
        __syncthreads();
    }
}

// dir = 1: decrease order; 0: increase order
template <typename T, bool dir, int blockSize, bool sorted>
__global__ void selectTopK(
    TensorInfo input,
    TensorInfo topK,
    TensorInfo indices,
    const int K,
    const int collapsedDims,
    const int sliceSize,
    const int inputSliceStride,
    const int topKSliceStride,
    const int indicesSliceStride)
{
    int inputSliceStart   = get_offset(blockIdx.x, collapsedDims, input);
    // if sorted, transform the output to coalesced slices
    int topKSliceStart    = sorted ? blockIdx.x * K : get_offset(blockIdx.x, collapsedDims, topK);
    int indicesSliceStart = sorted ? blockIdx.x * K : get_offset(blockIdx.x, collapsedDims, indices);

    // inc or dec hist every bin until reach K
    __shared__ int radix_hist[2 + blockSize / 32];
    int *smem = radix_hist;

    T *inputSlice     = (T *)input.data + inputSliceStart;
    T *topKSlice      = (T *)topK.data + topKSliceStart;
    int *indicesSlice = (int *)indices.data + indicesSliceStart;

    T fp_kthValue   = find_kth_value<T, dir>(smem, K, sliceSize, inputSlice, inputSliceStride);
    int writeStart  = 0;
    int activeWarps = 0;
    int tmpSize     = sliceSize;
    for (int off = threadIdx.x; off < Align(sliceSize, blockSize); off += blockSize) {
        int curSize         = tmpSize >= blockSize ? blockSize : tmpSize;
        activeWarps         = (curSize + 31) >> 5;
        bool inRange        = off < sliceSize;
        T value             = inRange ? inputSlice[off * inputSliceStride] : (T)0;
        unsigned int active = __ballot_sync(0xffffffff, inRange);

        bool flag;
        if (dir) {
            flag = inRange && Math<T, T, T>::gt(value, fp_kthValue);
        } else {
            flag = inRange && Math<T, T, T>::lt(value, fp_kthValue);
        }
        int index, blkTotal;
        prefix_scan(smem, active, activeWarps, flag, index, blkTotal);

        if (flag) {
            int topKOffset            = sorted ? (writeStart + index) : (writeStart + index) * topKSliceStride;
            int indexOffset           = sorted ? (writeStart + index) : (writeStart + index) * indicesSliceStride;
            topKSlice[topKOffset]     = value;
            indicesSlice[indexOffset] = off;
        }
        writeStart += blkTotal;
        // if tmpSize < 0, the loop breaks
        tmpSize -= blockSize;
    }

    int topKRemaining = K - writeStart;
    tmpSize           = sliceSize;
    for (int off = threadIdx.x; off < Align(sliceSize, blockSize); off += blockSize) {
        int curSize         = tmpSize >= blockSize ? blockSize : tmpSize;
        activeWarps         = (curSize + 31) >> 5;
        bool inRange        = off < sliceSize;
        T value             = inRange ? inputSlice[off * inputSliceStride] : (T)0;
        unsigned int active = __ballot_sync(0xffffffff, inRange);
        bool flag;
        flag = inRange && Math<T, T, T>::eq(value, fp_kthValue);
        int index, blkTotal;
        prefix_scan(smem, active, activeWarps, flag, index, blkTotal);

        if (flag) {
            int outputIndex = writeStart + index;
            if (outputIndex < K) {
                int topKOffset            = sorted ? outputIndex : outputIndex * topKSliceStride;
                int indexOffset           = sorted ? outputIndex : outputIndex * indicesSliceStride;
                topKSlice[topKOffset]     = value;
                indicesSlice[indexOffset] = off;
            }
        }
        if (topKRemaining < blkTotal) {
            break;
        }

        topKRemaining -= blkTotal;
        writeStart += blkTotal;
        tmpSize -= blockSize;
    }
}

template <typename KEY, typename VALUE, bool largest>
__device__ inline void swap(
    const bool isOdd,
    bool &valid1,
    KEY &value1,
    VALUE &index1,
    bool &valid2,
    KEY &value2,
    VALUE &index2)
{
    bool isLarge = (largest ^ Math<KEY, KEY, KEY>::lt(value1, value2) && valid1) ||
                   !valid2;
    if (isLarge == isOdd) {
        KEY tmpValue   = value1;
        VALUE tmpIndex = index1;
        bool tmpValid  = valid1;
        value1         = value2;
        index1         = index2;
        valid1         = valid2;
        value2         = tmpValue;
        index2         = tmpIndex;
        valid2         = tmpValid;
    }
}

template <typename KEY, typename VALUE, bool dir, int power2SortSize>
__global__ void bitonicSort(
    KEY *Key,
    VALUE *Value,
    const int sliceSize)
{
    __shared__ KEY smemTopk[power2SortSize];
    __shared__ VALUE smemIndices[power2SortSize];
    __shared__ bool smemValid[power2SortSize];

    KEY *topKSlice      = Key + blockIdx.x * sliceSize;
    VALUE *indicesSlice = Value + blockIdx.x * sliceSize;

    int tid       = threadIdx.x;
    int off1      = threadIdx.x;
    int off2      = threadIdx.x + power2SortSize / 2;
    bool inRange1 = off1 < sliceSize;
    bool inRange2 = off2 < sliceSize;
    KEY value1    = inRange1 ? topKSlice[off1] : (KEY)0;
    VALUE index1  = inRange1 ? indicesSlice[off1] : (VALUE)0;
    KEY value2    = inRange2 ? topKSlice[off2] : (KEY)0;
    VALUE index2  = inRange2 ? indicesSlice[off2] : (VALUE)0;

    smemTopk[off1]    = value1;
    smemIndices[off1] = index1;
    smemValid[off1]   = inRange1;
    smemTopk[off2]    = value2;
    smemIndices[off2] = index2;
    smemValid[off2]   = inRange2;
    __syncthreads();

#pragma unroll
    for (int size = 2; size < power2SortSize; size *= 2) {
        int oddSeg = (tid & (size / 2)) != 0;
#pragma unroll
        // sort each size
        for (int sub_size = size; sub_size > 1; sub_size /= 2) {
            int stride = sub_size / 2;
            int off    = (tid / stride) * sub_size + (tid & (stride - 1));

            bool inRange1 = smemValid[off];
            KEY value1    = smemTopk[off];
            VALUE index1  = smemIndices[off];
            bool inRange2 = smemValid[off + stride];
            KEY value2    = smemTopk[off + stride];
            VALUE index2  = smemIndices[off + stride];

            swap<KEY, VALUE, dir>(oddSeg,
                                  inRange1,
                                  value1,
                                  index1,
                                  inRange2,
                                  value2,
                                  index2);

            smemTopk[off]             = value1;
            smemIndices[off]          = index1;
            smemValid[off]            = inRange1;
            smemTopk[off + stride]    = value2;
            smemIndices[off + stride] = index2;
            smemValid[off + stride]   = inRange2;

            __syncthreads();
        }
    }

    // sort the whole power2SortSize
    for (int sub_size = power2SortSize; sub_size > 1; sub_size /= 2) {
        int stride = sub_size / 2;
        int off    = (tid / stride) * sub_size + (tid & (stride - 1));

        bool inRange1 = smemValid[off];
        KEY value1    = smemTopk[off];
        VALUE index1  = smemIndices[off];
        bool inRange2 = smemValid[off + stride];
        KEY value2    = smemTopk[off + stride];
        VALUE index2  = smemIndices[off + stride];

        swap<KEY, VALUE, dir>(false,
                              inRange1,
                              value1,
                              index1,
                              inRange2,
                              value2,
                              index2);

        smemTopk[off]             = value1;
        smemIndices[off]          = index1;
        smemValid[off]            = inRange1;
        smemTopk[off + stride]    = value2;
        smemIndices[off + stride] = index2;
        smemValid[off + stride]   = inRange2;

        __syncthreads();
    }

    inRange1 = smemValid[off1];
    value1   = smemTopk[off1];
    index1   = smemIndices[off1];
    inRange2 = smemValid[off2];
    value2   = smemTopk[off2];
    index2   = smemIndices[off2];
    if (inRange1) {
        topKSlice[off1]    = value1;
        indicesSlice[off1] = index1;
    }
    if (inRange2) {
        topKSlice[off2]    = value2;
        indicesSlice[off2] = index2;
    }
}

#define BLK_SORT_SIZE 1024
#define SIZE_PER_SCAN 1024

template <typename KEY, typename VALUE, bool dir>
void radix_sort(
    cudaStream_t stream,
    KEY *key,
    VALUE *value,
    int size,
    int sliceNum,
    unsigned int *convertKey,
    unsigned int *prefixData,
    unsigned int *tmpPrefix,
    unsigned int *keyBuf,
    VALUE *valueBuf);

// tempBuf:
//*dims
// convertKey: size*sizeof(unsigned int). zero if inplace
// keyBuf: convertKey size
// valueBuf: value size
// prefixData: SORT_RADIX_SIZE * blocks * sizeof(int)
// tmpPrefix: SORT_RADIX_SIZE * block_x* sizeof(uint)
template <typename KEY, typename VALUE, bool dir>
void sortInplace(
    cudaStream_t stream,
    KEY *Key,
    VALUE *Value,
    const int size,
    const int slices_num,
    void *temp_buffer,
    int64_t temp_buffer_bytes)
{
    const int blocks = slices_num;
    if (size == 1) {
    } else if (size <= 64) {
        bitonicSort<KEY, VALUE, dir, 64><<<blocks, 32, 0, stream>>>(Key, Value, size);
    } else if (size <= 128) {
        bitonicSort<KEY, VALUE, dir, 128><<<blocks, 64, 0, stream>>>(Key, Value, size);
    } else if (size <= 512) {
        bitonicSort<KEY, VALUE, dir, 512><<<blocks, 256, 0, stream>>>(Key, Value, size);
    } else {
        int new_blocks = (size + BLK_SORT_SIZE - 1) / BLK_SORT_SIZE;

        unsigned int *convert_key = (unsigned int *)temp_buffer;
        unsigned int *topk_buf    = (unsigned int *)(convert_key + slices_num * size);
        VALUE *indices_buf        = (VALUE *)(topk_buf + slices_num * size);
        unsigned int *prefix_data = (unsigned int *)(indices_buf + slices_num * size);
        unsigned int *tmp_prefix  = (unsigned int *)(prefix_data + slices_num * SORT_RADIX_SIZE * new_blocks);

        radix_sort<KEY, VALUE, dir>(stream, Key, Value, size, slices_num, convert_key, prefix_data, tmp_prefix, topk_buf, indices_buf);
    }
}

int collapse_dim(TensorInfo *param, int dim)
{
    int dimSize       = param->shape[dim];
    param->shape[dim] = 1;
    int cur           = -1;
    int p             = 0;
    for (; p < dim; p++) {
        if (param->shape[p] == 1)
            continue;
        cur++;
        param->shape[cur] = param->shape[p];
        p++;
        break;
    }
    for (; p < dim; p++) {
        if (param->shape[p] == 1)
            continue;
        cur++;
        param->shape[cur] = param->shape[p];
    }
    // after dim
    int markCur = cur;
    for (; p < param->dims; p++) {
        if (param->shape[p] == 1)
            continue;
        cur++;
        param->shape[cur] = param->shape[p];
    }

    param->strides[cur] = 1;
    for (int i = cur - 1; i > markCur; --i) {
        param->strides[i] = param->shape[i + 1] * param->strides[i + 1];
    }

    int sliceStride         = (dim == -1 || dim == param->dims - 1) ? 1 : param->shape[markCur + 1] * param->strides[markCur + 1];
    param->strides[markCur] = dimSize * sliceStride;

    for (int i = markCur - 1; i >= 0; --i) {
        param->strides[i] = param->shape[i + 1] * param->strides[i + 1];
    }

    param->dims = cur + 1;
    return sliceStride;
}

// bitWidth: 2 or 4 SORT_RADIX_SIZE
// bitPos: 0-30
template <bool dir, typename VALUE>
__global__ void radixSort(
    unsigned int *Key,
    VALUE *Value,
    int size,
    int bitPos,
    unsigned int *prefixData)
{
    __shared__ unsigned int s_cnt[SORT_RADIX_SIZE * BLK_SORT_SIZE / 32];
    __shared__ unsigned int s_key[BLK_SORT_SIZE];
    __shared__ VALUE s_value[BLK_SORT_SIZE];

    Key += blockIdx.y * size;
    Value += blockIdx.y * size;

    s_key[threadIdx.x]   = 0;
    s_value[threadIdx.x] = 0;
    if (threadIdx.x < SORT_RADIX_SIZE * BLK_SORT_SIZE / 32)
        s_cnt[threadIdx.x] = 0;
    __syncthreads();

    int lane        = threadIdx.x & 31;
    int warpId      = threadIdx.x >> 5;
    int activeWarps = (blockIdx.x == gridDim.x - 1) ? DivUp((size & (BLK_SORT_SIZE - 1)), 32) : BLK_SORT_SIZE / 32;
    if (activeWarps == 0)
        activeWarps = 32;
    int64_t tid                     = blockIdx.x * blockDim.x + threadIdx.x;
    bool inRange                    = tid < size;
    unsigned int active             = __ballot_sync(0xffffffff, inRange);
    const unsigned int lane_mask_lt = ~(0xffffffff << (lane));
    if (tid - lane < size) {
        VALUE value         = inRange ? Value[tid] : (VALUE)0;
        unsigned int intKey = inRange ? Key[tid] : 0;
        unsigned int keyRadix;
        asm("bfe.u32 %0, %1, %2, %3;"
            : "=r"(keyRadix)
            : "r"(intKey), "r"(bitPos), "r"(SORT_RADIX_BITS));
        int radixPrefix = 0;
        for (int i = dir * (SORT_RADIX_SIZE - 1);
             i != (1 - dir) * SORT_RADIX_SIZE + dir * (-1);
             i += dir * (-1) + 1 - dir) {
            bool flag           = inRange && (keyRadix == i);
            unsigned int ballot = __ballot_sync(active, flag);
            int warpCnt         = __popc(ballot);
            int lanePrefix      = __popc(ballot & lane_mask_lt);
            int warpPrefix      = 0;
            if (inRange && lane == 0) {
                s_cnt[i * BLK_SORT_SIZE / 32 + warpId] = warpCnt;
            }
            __syncthreads();

            // prefix sum in warp
            if (threadIdx.x < 32) {
                warpCnt             = s_cnt[i * BLK_SORT_SIZE / 32 + threadIdx.x];
                unsigned int prefix = warpCnt;
                for (int j = 1; j < 32; j <<= 1) {
                    warpCnt = __shfl_up_sync(0xffffffff, prefix, j, 32);
                    if (threadIdx.x >= j) {
                        prefix += warpCnt;
                    }
                }
                s_cnt[i * BLK_SORT_SIZE / 32 + threadIdx.x] = prefix;
            }
            __syncthreads();
            if (inRange && warpId > 0) {
                warpPrefix = s_cnt[i * BLK_SORT_SIZE / 32 + warpId - 1];
            }

            if (flag) {
                s_key[radixPrefix + warpPrefix + lanePrefix]   = intKey;
                s_value[radixPrefix + warpPrefix + lanePrefix] = value;
            }

            radixPrefix += s_cnt[i * BLK_SORT_SIZE / 32 + activeWarps - 1];
            __syncthreads(); // WAR
        }
        if (threadIdx.x == 0) {
            for (int i = 0; i < SORT_RADIX_SIZE; i++) {
                prefixData[blockIdx.y * gridDim.x * SORT_RADIX_SIZE + i * gridDim.x + blockIdx.x] =
                    s_cnt[i * BLK_SORT_SIZE / 32 + activeWarps - 1];
            }
        }

        intKey = s_key[threadIdx.x];
        value  = s_value[threadIdx.x];
        if (inRange) {
            Key[tid]   = intKey;
            Value[tid] = value;
        }
    }
}

template <typename T>
__device__ T scanInWarp(T value, int lane)
{
    T lanePrefix = value;
    for (int i = 1; i < 32; i <<= 1) {
        value = __shfl_up_sync(0xffffffff, lanePrefix, i, 32);
        if (lane >= i) {
            lanePrefix += value;
        }
    }
    return lanePrefix;
}
//#define SIZE_PER_SCAN 1024
template <typename T>
__global__ void prefixSum(
    T *prefixData,
    const int size,
    const int blkScanSize,
    T *blkTotal)
{
    __shared__ T warp_cnt[SIZE_PER_SCAN >> 5];
    __shared__ T lane_prefix[SIZE_PER_SCAN];
    int lane    = threadIdx.x & 31;
    int warpId  = threadIdx.x >> 5;
    int64_t off = blockIdx.x * blkScanSize + threadIdx.x;
    prefixData += (blockIdx.z * SORT_RADIX_SIZE + blockIdx.y) * size;
    blkTotal += blockIdx.z * SORT_RADIX_SIZE * gridDim.x +
                blockIdx.y * gridDim.x + blockIdx.x;

    T subScanPrefix = (T)0;
    for (int iterOff = 0; iterOff < blkScanSize; iterOff += SIZE_PER_SCAN) {
        bool inRange = (off + iterOff < size);
        T data       = inRange ? prefixData[off + iterOff] : 0;
        T lanePrefix;

        lanePrefix = data;
        for (int i = 1; i < 32; i <<= 1) {
            data = __shfl_up_sync(0xffffffff, lanePrefix, i, 32);
            if (lane >= i) {
                lanePrefix += data;
            }
        }
        int stsOff = threadIdx.x + 1;
        if (lane == 31) {
            warp_cnt[warpId] = lanePrefix;
            stsOff -= 32;
            lanePrefix = 0;
        }
        __syncthreads();
        lane_prefix[stsOff] = lanePrefix;

        T warpPrefix = 0;
        if (threadIdx.x < (SIZE_PER_SCAN >> 5)) {
            data  = warp_cnt[threadIdx.x];
            T sum = data;
            for (int i = 1; i < 32; i <<= 1) {
                data = __shfl_up_sync(0xffffffff, sum, i, 32);
                if (threadIdx.x >= i) {
                    sum += data;
                }
            }
            warp_cnt[threadIdx.x] = sum;
        }
        __syncthreads();

        lanePrefix = lane_prefix[threadIdx.x];
        if (warpId > 0) {
            warpPrefix = warp_cnt[warpId - 1];
        }

        T prefix = subScanPrefix + warpPrefix + lanePrefix;
        if (inRange) {
            prefixData[off + iterOff] = prefix;
        }
        subScanPrefix += warp_cnt[(SIZE_PER_SCAN >> 5) - 1];
    }
    // blk scan total
    if (threadIdx.x == 0) {
        blkTotal[0] = subScanPrefix;
    }
}
template <typename T>
__global__ void finalPrefixSum(
    T *prefixData,
    const int size,
    const int blkScanSize,
    T *blkTotal)
{
    int batchId = blockIdx.z * SORT_RADIX_SIZE + blockIdx.y;
    int64_t off = blockIdx.x * blkScanSize + threadIdx.x;
    prefixData += batchId * size;

    T blkPrefix = 0;
    for (int i = 0; i < blockIdx.x; i++) {
        blkPrefix += blkTotal[batchId * gridDim.x + i];
    }

    for (int iterOff = 0; iterOff < blkScanSize; iterOff += SIZE_PER_SCAN) {
        bool inRange = (off + iterOff < size);
        T data       = inRange ? prefixData[off + iterOff] : 0;
        data += blkPrefix;
        if (inRange) {
            prefixData[off + iterOff] = data;
        }
    }
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
        blkTotal[batchId * gridDim.x + blockIdx.x] =
            blkPrefix + blkTotal[batchId * gridDim.x + blockIdx.x];
    }
}

template <bool dir, typename VALUE>
__global__ void interBlkSort(
    unsigned int *outKey,
    VALUE *outValue,
    unsigned int *Key,
    VALUE *Value,
    unsigned int size,
    unsigned int *prefixData,
    unsigned int *radixTotal,
    unsigned int totalPos,
    unsigned int bitPos)
{
    prefixData += blockIdx.y * SORT_RADIX_SIZE * gridDim.x;
    radixTotal += blockIdx.y * SORT_RADIX_SIZE * totalPos;
    Key += blockIdx.y * size;
    Value += blockIdx.y * size;
    outKey += blockIdx.y * size;
    outValue += blockIdx.y * size;

    __shared__ unsigned int s_cnt[SORT_RADIX_SIZE * BLK_SORT_SIZE / 32];
    int lane    = threadIdx.x & 31;
    int warpId  = threadIdx.x / 32;
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < SORT_RADIX_SIZE * BLK_SORT_SIZE / 32)
        s_cnt[threadIdx.x] = 0;
    __syncthreads();
    int activeWarps = (blockIdx.x == gridDim.x - 1) ? DivUp((size & (BLK_SORT_SIZE - 1)), 31) : BLK_SORT_SIZE / 32;
    if (activeWarps == 0)
        activeWarps = 32;

    bool inRange                    = tid < size;
    unsigned int active             = __ballot_sync(0xffffffff, inRange);
    const unsigned int lane_mask_lt = ~(0xffffffff << (lane));
    if (tid - lane < size) {
        unsigned int intKey      = inRange ? Key[tid] : 0;
        VALUE value              = inRange ? Value[tid] : (VALUE)0;
        unsigned int radixPrefix = 0;

        unsigned int newOff;
        unsigned int keyRadix;
        asm("bfe.u32 %0, %1, %2, %3;"
            : "=r"(keyRadix)
            : "r"(intKey), "r"(bitPos), "r"(SORT_RADIX_BITS));
        for (int i = dir * (SORT_RADIX_SIZE - 1);
             i != (1 - dir) * SORT_RADIX_SIZE + dir * (-1);
             i += dir * (-1) + 1 - dir) {
            unsigned int blkPrefix  = prefixData[i * gridDim.x + blockIdx.x];
            bool flag               = inRange && (keyRadix == i);
            unsigned int ballot     = __ballot_sync(active, flag);
            unsigned int lanePrefix = __popc(lane_mask_lt & ballot);
            unsigned int warpCnt    = __popc(ballot);
            unsigned int warpPrefix = 0;

            if (inRange && lane == 0) {
                s_cnt[i * BLK_SORT_SIZE / 32 + warpId] = warpCnt;
            }
            __syncthreads();
            if (threadIdx.x < 32) {
                warpCnt             = s_cnt[i * BLK_SORT_SIZE / 32 + threadIdx.x];
                unsigned int prefix = warpCnt;
                for (int j = 1; j < 32; j <<= 1) {
                    warpCnt = __shfl_up_sync(0xffffffff, prefix, j, 32);
                    if (threadIdx.x >= j) {
                        prefix += warpCnt;
                    }
                }
                s_cnt[i * BLK_SORT_SIZE / 32 + threadIdx.x] = prefix;
            }
            __syncthreads();
            if (inRange && warpId > 0) {
                warpPrefix = s_cnt[i * BLK_SORT_SIZE / 32 + warpId - 1];
            }
            __syncthreads();

            if (flag) {
                newOff = radixPrefix + blkPrefix + warpPrefix + lanePrefix;
            }
            __syncthreads();
            radixPrefix += radixTotal[i * totalPos + totalPos - 1];
        }

        if (inRange) {
            outKey[newOff]   = intKey;
            outValue[newOff] = value;
        }
    }
}

// for fp16, we can apply short int *outKey
template <typename KEY>
__global__ void convert(KEY *Key, unsigned int *outKey, int size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;
    KEY key             = Key[tid];
    unsigned int intKey = convert2u<KEY>(key);

    outKey[tid] = intKey;
}
template <typename KEY>
__global__ void reverse_convert(unsigned int *Key, KEY *outKey, int size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    unsigned int intKey = Key[tid];
    KEY key             = convertu2<KEY>(intKey);
    outKey[tid]         = key;
}

// tempBuf:
//*dims
// convertKey: size*sizeof(unsigned int). zero if inplace
// keyBuf: convertKey size
// valueBuf: value size
// prefixData: SORT_RADIX_SIZE * blocks * sizeof(int)
// tmpPrefix: SORT_RADIX_SIZE * block_x* sizeof(uint)
template <typename KEY, typename VALUE, bool largest>
void radix_sort(
    cudaStream_t stream,
    KEY *key,
    VALUE *value,
    int size,
    int sliceNum,
    unsigned int *convertKey,
    unsigned int *prefixData,
    unsigned int *tmpPrefix,
    unsigned int *keyBuf,
    VALUE *valueBuf)
{
    convert<KEY><<<DivUp(sliceNum * size, 1024), 1024, 0, stream>>>(key, convertKey, sliceNum * size);
    int blocks = DivUp(size, BLK_SORT_SIZE);

    constexpr int MAX_BLKS = 64;
    int prefixSize         = blocks;
    int blkScanSize        = max(DivUp(prefixSize, MAX_BLKS),
                          SIZE_PER_SCAN);
    unsigned int block_x   = DivUp(blocks, blkScanSize);

    unsigned int *keyIn, *keyOut;
    VALUE *valueIn, *valueOut;
    keyIn    = convertKey;
    valueIn  = value;
    keyOut   = keyBuf;
    valueOut = valueBuf;

    dim3 sort_grid   = dim3(blocks, sliceNum, 1);
    dim3 prefix_grid = dim3(block_x, SORT_RADIX_SIZE, sliceNum);
    dim3 final_grid  = dim3(block_x, SORT_RADIX_SIZE, sliceNum);
    for (unsigned pos = 0; pos <= 8 * sizeof(KEY) - SORT_RADIX_BITS; pos += SORT_RADIX_BITS) {
        radixSort<largest, VALUE><<<sort_grid, BLK_SORT_SIZE, 0, stream>>>(keyIn, valueIn, size, pos, prefixData);

        prefixSum<unsigned int><<<prefix_grid, SIZE_PER_SCAN, 0, stream>>>(prefixData, blocks, blkScanSize, tmpPrefix);
        if (block_x > 1) {
            finalPrefixSum<unsigned int><<<final_grid, SIZE_PER_SCAN, 0, stream>>>(prefixData, blocks, blkScanSize, tmpPrefix);
        }
        interBlkSort<largest, VALUE><<<sort_grid, BLK_SORT_SIZE, 0, stream>>>(keyOut, valueOut, keyIn, valueIn, size, prefixData, tmpPrefix, block_x, pos);

        unsigned int *tmpk = keyIn;
        VALUE *tmpv        = valueIn;

        keyIn    = keyOut;
        keyOut   = tmpk;
        valueIn  = valueOut;
        valueOut = tmpv;
    }
    if (keyIn != convertKey) {
        cudaMemcpyAsync(value, valueOut, size * sizeof(VALUE), cudaMemcpyDeviceToDevice, stream);
    }

    reverse_convert<KEY><<<DivUp(sliceNum * size, 1024), 1024, 0, stream>>>(keyIn, key, sliceNum * size);
}

int64_t PPLTopKGetTempBufferSize(
    const ppl::nn::TensorShape *indices_shape,
    const int K,
    int dim_k,
    bool sorted)
{
    if (sorted == false)
        return 0;

    if (dim_k == -1) {
        dim_k = indices_shape->GetDimCount() - 1;
    }
    int slices_num = 1;
    for (unsigned int i = 0; i < indices_shape->GetDimCount(); i++) {
        if (i != (unsigned int)dim_k)
            slices_num *= indices_shape->GetDim(i);
    }

    int64_t total_size = 0;
    // keyBuf
    total_size += slices_num * K * sizeof(unsigned int);
    // valueBuf unsigned int
    total_size += slices_num * K * sizeof(indices_shape->GetDataType());
    // max bitonic sort size
    if (K <= 512)
        return total_size;

    // determined by GPU devices, SMs number
    constexpr int MAX_BLKS = 64;
    int new_blocks         = (K + BLK_SORT_SIZE - 1) / BLK_SORT_SIZE;
    int prefixSize         = new_blocks;
    int blkScanSize        = max(DivUp(prefixSize, MAX_BLKS), SIZE_PER_SCAN);
    unsigned int block_x   = DivUp(new_blocks, blkScanSize);
    // convertKey
    total_size += slices_num * K * sizeof(unsigned int);
    // prefixData
    total_size += slices_num * SORT_RADIX_SIZE * new_blocks * sizeof(unsigned int);
    // tmpPrefix
    total_size += slices_num * SORT_RADIX_SIZE * block_x * sizeof(unsigned int);

    return total_size;
}

template <typename T>
__global__ void transpose(
    const T *input,
    T *output,
    const int batch,
    const int input_h,
    const int input_w)
{
    __shared__ T smem[32][33];
    int i_h         = blockIdx.y * 32 + threadIdx.y;
    int i_w         = blockIdx.x * 32 + threadIdx.x;
    int o_w         = blockIdx.y * 32 + threadIdx.x;
    int o_h         = blockIdx.x * 32 + threadIdx.y;
    bool inBound0   = i_h < input_h && i_w < input_w;
    int64_t index   = (blockIdx.z * input_h + i_h) * input_w + i_w;
    bool inBound1   = o_h < input_w && o_w < input_h;
    int64_t o_index = (blockIdx.z * input_w + o_h) * input_h + o_w;

    T value                        = inBound0 ? input[index] : (T)0;
    smem[threadIdx.x][threadIdx.y] = value;
    __syncthreads();
    value = smem[threadIdx.y][threadIdx.x];

    if (inBound1) {
        output[o_index] = value;
    }
}

template <typename T, typename ID>
void topKGpuImpl(
    const cudaStream_t &stream,
    const int K,
    int dim,
    TensorInfo inputInfo,
    TensorInfo topKInfo,
    TensorInfo indicesInfo,
    void *temp_buffer,
    int64_t temp_buffer_bytes,
    const bool largest = true,
    const bool sorted  = true)
{
    bool is_trans = false;
    int batch     = 1;
    int trans_h = 1, trans_w = 1;
    if (dim == -1) {
        dim = inputInfo.dims - 1;
    }
    if (dim != inputInfo.dims - 1) {
        is_trans = true;
        trans_w  = K;
        for (int i = 0; i < dim; i++) {
            batch *= topKInfo.shape[i];
        }
        for (int i = dim + 1; i < topKInfo.dims; i++) {
            trans_h *= topKInfo.shape[i];
        }
    }
    int sliceSize          = inputInfo.shape[dim];
    // collapse dim_k and dim which is size of 1
    int inputSliceStride   = collapse_dim(&inputInfo, dim);
    int topKSliceStride    = collapse_dim(&topKInfo, dim);
    int indicesSliceStride = collapse_dim(&indicesInfo, dim);

    int blocks = 1;
    for (int i = 0; i < inputInfo.dims; ++i) {
        blocks *= inputInfo.shape[i];
    }

#define POSTLOG_TRANSPOSE()                                                                                                    \
    {                                                                                                                          \
        if (is_trans) {                                                                                                        \
            int trans_size    = batch * trans_h * trans_w;                                                                     \
            dim3 block_size   = dim3(32, 32, 1);                                                                               \
            dim3 grid         = dim3(DivUp(trans_w, 32),                                                                       \
                             DivUp(trans_h, 32),                                                                       \
                             batch);                                                                                   \
            T *trans_topk     = reinterpret_cast<T *>(temp_buffer);                                                            \
            ID *trans_indices = reinterpret_cast<ID *>(trans_topk + trans_size);                                               \
                                                                                                                               \
            transpose<T><<<grid, block_size, 0, stream>>>((T *)topKInfo.data, trans_topk, batch, trans_h, trans_w);            \
            transpose<ID><<<grid, block_size, 0, stream>>>((ID *)indicesInfo.data, trans_indices, batch, trans_h, trans_w);    \
            cudaMemcpyAsync((T *)topKInfo.data, trans_topk, trans_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);         \
            cudaMemcpyAsync((ID *)indicesInfo.data, trans_indices, trans_size * sizeof(ID), cudaMemcpyDeviceToDevice, stream); \
        }                                                                                                                      \
    }
    constexpr int BLK_SIZE = 1024;
    if (largest) {
        if (sorted) { // index is int32 by default
            selectTopK<T, 1, BLK_SIZE, 1><<<blocks, BLK_SIZE, 0, stream>>>(inputInfo, topKInfo, indicesInfo, K, inputInfo.dims, sliceSize, inputSliceStride, topKSliceStride, indicesSliceStride);
            sortInplace<T, ID, 1>(stream, (T *)topKInfo.data, (ID *)indicesInfo.data, K, blocks, temp_buffer, temp_buffer_bytes);
            // transpose
            POSTLOG_TRANSPOSE();
        } else {
            selectTopK<T, 1, BLK_SIZE, 0><<<blocks, BLK_SIZE, 0, stream>>>(inputInfo, topKInfo, indicesInfo, K, inputInfo.dims, sliceSize, inputSliceStride, topKSliceStride, indicesSliceStride);
        }
    } else {
        if (sorted) {
            selectTopK<T, 0, BLK_SIZE, 1><<<blocks, BLK_SIZE, 0, stream>>>(inputInfo, topKInfo, indicesInfo, K, inputInfo.dims, sliceSize, inputSliceStride, topKSliceStride, indicesSliceStride);
            sortInplace<T, ID, 0>(stream, (T *)topKInfo.data, (ID *)indicesInfo.data, K, blocks, temp_buffer, temp_buffer_bytes);
            // transpose
            POSTLOG_TRANSPOSE();
        } else {
            selectTopK<T, 0, BLK_SIZE, 0><<<blocks, BLK_SIZE, 0, stream>>>(inputInfo, topKInfo, indicesInfo, K, inputInfo.dims, sliceSize, inputSliceStride, topKSliceStride, indicesSliceStride);
        }
    }
#undef POSTLOG_TRANSPOSE
}

ppl::common::RetCode PPLCUDATopKForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape *input_shape,
    const void *input,
    ppl::nn::TensorShape *topk_shape,
    void *topk,
    ppl::nn::TensorShape *indices_shape,
    int *indices,
    void *temp_buffer,
    int64_t temp_buffer_bytes,
    int K,
    int dim_k,
    const bool largest,
    const bool sorted)
{
    TensorInfo input_info(input_shape, input);
    TensorInfo topk_info(topk_shape, topk);
    TensorInfo indices_info(indices_shape, indices);
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        topKGpuImpl<float, int>(stream, K, dim_k, input_info, topk_info, indices_info, temp_buffer, temp_buffer_bytes, largest, sorted);
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        topKGpuImpl<__half, int>(stream, K, dim_k, input_info, topk_info, indices_info, temp_buffer, temp_buffer_bytes, largest, sorted);
    }

    return ppl::common::RC_SUCCESS;
}
