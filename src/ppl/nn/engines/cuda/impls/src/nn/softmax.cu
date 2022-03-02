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

#include "cudakernel/nn/softmax.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/reduce/reduce.h"
#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/unary/exp.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "cudakernel/common/common.cuh"

uint64_t PPLSoftmaxGetTempBufferSize(
    const ppl::nn::TensorShape* input_shape,
    int axis)
{
    int N = input_shape->GetElementsToDimensionIncludingPadding(axis);
    return N * ppl::common::GetSizeOfDataType(input_shape->GetDataType());
}

ppl::common::RetCode PPLCUDASoftmaxForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* temp_buffer,
    int axis)
{
    int N = input_shape->GetElementsToDimensionIncludingPadding(axis);
    int D = input_shape->GetElementsFromDimensionIncludingPadding(axis);
    // reduce max
    PPLReduceDimDes reduce_desc(1, D, N);
    ReduceParam reduce_max = ReduceMax;
    void* max_sum_output   = temp_buffer;
    ppl::nn::TensorShape max_sum_shape(*input_shape);
    max_sum_shape.SetDim(0, N);
    max_sum_shape.SetDim(1, 1);
    max_sum_shape.SetDimCount(2);
    
    auto status = PPLCUDAReduceForwardImp(stream, reduce_max, reduce_desc, input_shape, input, &max_sum_shape, max_sum_output);
    // sub
    ppl::nn::TensorShape nd_shape(*input_shape);
    nd_shape.SetDim(0, N);
    nd_shape.SetDim(1, D);
    nd_shape.SetDimCount(2);
    status = PPLCUDAArithMeticSubForwardImp(stream, &nd_shape, input, &max_sum_shape, max_sum_output, &nd_shape, output,1,1,1);
    // exp
    status                 = PPLCUDAExpForwardImp(stream, &nd_shape, output, &nd_shape, output);
    // reduce sum
    ReduceParam reduce_sum = ReduceSum;
    status = PPLCUDAReduceForwardImp(stream, reduce_sum, reduce_desc, &nd_shape, output, &max_sum_shape, max_sum_output);
    //div
    status = PPLCUDAArithMeticDivForwardImp(stream, &nd_shape, output, &max_sum_shape, max_sum_output, &nd_shape, output,1,1,1);
    return status;
}

#define CREATE_SOFTMAXSCORE_KERNEL_BOOL32(mask_type, buffer)                                            \
    template <typename Tin, typename MaskT, typename Tout,                                              \
                typename TCompute = float>                                                              \
    __global__ void SoftmaxScoreKernel32##mask_type(                                                    \
        const Tin* in, const MaskT* key_padding_mask, Tout* out,                                        \
        const int B, const int H, const int T) {                                                        \
            auto cur_in = in + blockIdx.x * T;                                                          \
            auto cur_out = out + blockIdx.x * T;                                                        \
            auto cur_mask = key_padding_mask + buffer;                                                  \
            TCompute log_sum = CudaLogZero<TCompute>();                                                  \
            for (auto tid = threadIdx.x; tid < T; tid += blockDim.x) {                                  \
                TCompute maskv = (TCompute) static_cast<float>(_Ldg(cur_mask + tid));                   \
                log_sum =                                                                               \
                        _LogAdd((TCompute)__ldg(cur_in + tid) * ((TCompute)1.0f - maskv) +              \
                                CudaLogZero<TCompute>() * maskv, log_sum);                              \
            }                                                                                           \
            log_sum = WarpReduceLogAddSum(log_sum);                                                     \
            for (auto tid = threadIdx.x; tid < T; tid += blockDim.x) {                                  \
                TCompute maskv = (TCompute) static_cast<float>(_Ldg(cur_mask + tid));                   \
                cur_out[tid] = (Tout)(_Exp((TCompute)__ldg(cur_in + tid) - log_sum) *   \
                                (TCompute)(1.0f - maskv));                                              \
            }                                                                                           \
        }
CREATE_SOFTMAXSCORE_KERNEL_BOOL32(Mask0, blockIdx.x * T)
CREATE_SOFTMAXSCORE_KERNEL_BOOL32(Mask1, blockIdx.x / H * T)
CREATE_SOFTMAXSCORE_KERNEL_BOOL32(Mask2, blockIdx.x / (H * T) * T)
CREATE_SOFTMAXSCORE_KERNEL_BOOL32(Mask3, blockIdx.x / (B * H) * T)




template<typename Tin, typename Tout, typename TCompute = float>
__global__ void SoftmaxScoreKernel32(const Tin* in, Tout* out, const int T) {
    auto cur_in = in + blockIdx.x * T;
    auto cur_out = out + blockIdx.x * T;
    // reduce log sum
    TCompute log_sum = CudaLogZero<TCompute>();
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        log_sum = _LogAdd((TCompute)__ldg(cur_in + tid), log_sum);
    }
    log_sum = WarpReduceLogAddSum(log_sum);
    for(auto tid = threadIdx.x; tid < T; tid += blockDim.x) {
        cur_out[tid] = _Exp((TCompute)__ldg(cur_in + tid) - log_sum);
    }
}


/*
par: 
    in & out : [BHTT]
    key_padding_mask : [B, H, T, T] or [B, 1, T, T] or [B, 1, 1, T], or [1, 1, T, T]
*/
template<typename Tin, typename MaskT, typename Tout>
ppl::common::RetCode PPLCUDAFastSoftmaxForwardImp(
    cudaStream_t stream,
    const Tin* input,
    Tout* output,
    const MaskT* key_padding_mask,
    const int mask_type,
    const int B,
    const int H,
    const int T)
{
    dim3 grid(B * H * T, 1, 1);
    if (key_padding_mask != nullptr) {
        dim3 block(32);
        if(mask_type == 0) {
            SoftmaxScoreKernel32Mask0<Tin, MaskT, Tout, float>
                <<<grid, block, 0, stream>>>(input, key_padding_mask, output, B, H, T);
        } else if(mask_type == 1) {
            SoftmaxScoreKernel32Mask1<Tin, MaskT, Tout, float>
                <<<grid, block, 0, stream>>>(input, key_padding_mask, output, B, H, T);
        } else if(mask_type == 2) {
            SoftmaxScoreKernel32Mask2<Tin, MaskT, Tout, float>
                <<<grid, block, 0, stream>>>(input, key_padding_mask, output, B, H, T);
        } else if(mask_type == 3) {
            SoftmaxScoreKernel32Mask3<Tin, MaskT, Tout, float>
                <<<grid, block, 0, stream>>>(input, key_padding_mask, output, B, H, T);
        }
    } else {
        dim3 block(32);
        SoftmaxScoreKernel32<Tin, Tout, float>
            <<<grid, block, 0, stream>>>(input, output, T);
    }
    return ppl::common::RC_SUCCESS;
}

#define CREATE_SOFTMAXSCORE_FUN(Tin, MaskT, Tout)                                               \
    template ppl::common::RetCode PPLCUDAFastSoftmaxForwardImp<Tin, MaskT, Tout>(               \
        cudaStream_t stream, const Tin* input, Tout* output, const MaskT* key_padding_mask,     \
        const int mask_type, const int B, const int H, const int T);                            \

CREATE_SOFTMAXSCORE_FUN(float, bool, float)
CREATE_SOFTMAXSCORE_FUN(double, bool, double)
CREATE_SOFTMAXSCORE_FUN(half, bool, half)

ppl::common::RetCode PPLCUDAFastSoftmax(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    const void* key_padding_mask,
    const int mask_type) {
        if(output_shape->GetDataType() == 5)  {
            return PPLCUDAFastSoftmaxForwardImp<half, bool, half>(stream, (const half*)input, (half*)output, (const bool*)key_padding_mask, mask_type, input_shape->GetDim(0), 1, input_shape->GetDim(1));
        } else if(output_shape->GetDataType() == 6) {
            return PPLCUDAFastSoftmaxForwardImp<float, bool, float>(stream, (const float*)input, (float*)output, (const bool*)key_padding_mask, mask_type, input_shape->GetDim(0), 1, input_shape->GetDim(1));
        }
        return ppl::common::RC_UNSUPPORTED;
    }