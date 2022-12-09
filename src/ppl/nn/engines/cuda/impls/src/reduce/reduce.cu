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

#include "cudakernel/reduce/reduce.h"
#include "cudakernel/reduce/reduce_kernel.h"
#include "ppl/nn/engines/cuda/impls/src/reformat/cvt_int8_float.cuh"

__global__ void CudaSetInitVal(
    half* output,
    half initval,
    int64_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    float* output,
    float initval,
    int64_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    int64_t* output,
    int64_t initval,
    int64_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    int32_t* output,
    int32_t initval,
    int32_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
__global__ void CudaSetInitVal(
    int8_t* output,
    int8_t initval,
    int8_t size)
{
    int tid     = blockIdx.x * gridDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    output[tid] = initval;
}
template <typename T>
void SetInitVal(
    void* output,
    T initval,
    int64_t size)
{
    dim3 blockDim(32, 32);
    dim3 gridDim(DivUp(32 * 32, size));
    CudaSetInitVal<<<gridDim, blockDim>>>((T *)output, initval, size);
    cudaDeviceSynchronize();
}

template <typename srcT>
__global__ void rescale_from_int8_to_int8(int32_t num_elems,
    const srcT* input, int8_t* output, ppl::nn::cuda::QuantParamCuda qparam) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= num_elems) return;
    float inter_val = (float)(input[tid] - qparam.i_zero_point) * qparam.i_step;
    output[tid] = _float2int8(inter_val, qparam.o_step, qparam.o_zero_point);
}

template <typename srcT>
ppl::common::RetCode PPLRescaleInt82Int8(cudaStream_t stream, const ppl::nn::TensorShape* output_shape,
    const srcT* input, int8_t* output, const ppl::nn::cuda::QuantParamCuda* qparam) {
    int num_elems = output_shape->CalcElementsExcludingPadding();
    int block_size = 256;
    int grid_size = (num_elems + block_size - 1) / block_size;
    rescale_from_int8_to_int8<srcT><<<grid_size, block_size, 0, stream>>>(num_elems,
    input, output, *qparam);
    return ppl::common::RC_SUCCESS;
}

std::pair<dim3, dim3> ComputeKernelConfigure(
    ReduceParam param,
    ReduceMode mode,
    int64_t& num_elements,
    bool& multi_block,
    PPLReduceDimDes& des)
{
    switch (mode) {
        case ReduceAll:
            return ComputeReduceAllConfigure(param, mode, num_elements, multi_block, des);
        case ReduceRow:
            return ComputeReduceRowConfigure(param, mode, num_elements, multi_block, des);
        case ReduceCol:
            return ComputeReduceColConfigure(param, mode, num_elements, multi_block, des);
        default:
            return ComputeReduceAllConfigure(param, mode, num_elements, multi_block, des);
    }
}

template <class Operator>
ppl::common::RetCode PPLCUDAReduceOPImp(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::srctype in_type;
    typedef typename Operator::dsttype dst_type;
    Operator op((in_type *)input, (dst_type *)output);
    ReduceMode mode      = pplGetReduceMode(des);
    bool multi_block     = false;
    int64_t num_elements = 1;
    auto configure       = ComputeKernelConfigure(param, mode, num_elements, multi_block, des);
    dst_type initval     = static_cast<dst_type>(op.InitVal());
#define CASE(Mode)                                                                                                                         \
    case Mode:                                                                                                                             \
        if (multi_block) {                                                                                                                 \
            SetInitVal(output, initval, des.n_inner *des.n_outer);                                                                         \
            ppl_reduce<acc_type, Operator, BLOCKSIZE, true, (int)Mode><<<configure.second, configure.first, 0, stream>>>(op, des, param);  \
        } else                                                                                                                             \
            ppl_reduce<acc_type, Operator, BLOCKSIZE, false, (int)Mode><<<configure.second, configure.first, 0, stream>>>(op, des, param); \
        break;

    switch (mode) {
        CASE(ReduceAll)
        CASE(ReduceRow)
        CASE(ReduceCol)
        default:
            return ppl::common::RC_SUCCESS;
    }
#undef CASE
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode PPLCUDAReduceForwardImp(
    cudaStream_t stream,
    ReduceParam param,
    PPLReduceDimDes des,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output,
    void* tmp_buffer,
    const ppl::nn::cuda::QuantParamCuda* qparam)
{
#define CASEFP16(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEFP32(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEINT64(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEINT32(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        return PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, output);
#define CASEINT8(Mode, OP, Tin, Tout, Tacc) \
    case Mode:                              \
        status = PPLCUDAReduceOPImp<OP<Tin, Tout, Tacc>>(stream, param, des, input_shape, input, output_shape, tmp_buffer);

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        switch (param) {
            CASEFP16(ReduceSum, SumOp, half, half, float)
            CASEFP16(ReduceProd, ProdOp, half, half, float)
            CASEFP16(ReduceMean, SumOp, half, half, float)
            CASEFP16(ReduceMax, MaxOp, half, half, half)
            CASEFP16(ReduceMin, MinOp, half, half, half)
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
        switch (param) {
            CASEFP32(ReduceSum, SumOp, float, float, float)
            CASEFP32(ReduceProd, ProdOp, float, float, float)
            CASEFP32(ReduceMean, SumOp, float, float, float)
            CASEFP32(ReduceMax, MaxOp, float, float, float)
            CASEFP32(ReduceMin, MinOp, float, float, float)
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT64) {
        switch (param) {
            CASEINT64(ReduceSum, SumOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceProd, ProdOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceMean, SumOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceMax, MaxOp, int64_t, int64_t, int64_t)
            CASEINT64(ReduceMin, MinOp, int64_t, int64_t, int64_t)
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT32) {
        switch (param) {
            CASEINT32(ReduceSum, SumOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceProd, ProdOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceMean, SumOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceMax, MaxOp, int32_t, int32_t, int32_t)
            CASEINT32(ReduceMin, MinOp, int32_t, int32_t, int32_t)
            default:
                return ppl::common::RC_UNSUPPORTED;
        }
    } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        ppl::common::RetCode status = ppl::common::RC_SUCCESS;
        switch (param) {
            CASEINT8(ReduceSum, SumOp, int8_t, float, float)
            CASEINT8(ReduceProd, ProdOp, int8_t, float, float)
            CASEINT8(ReduceMean, SumOp, int8_t, float, float)
            CASEINT8(ReduceMax, MaxOp, int8_t, int8_t, int8_t)
            CASEINT8(ReduceMin, MinOp, int8_t, int8_t, int8_t)
            default:
                status =  ppl::common::RC_UNSUPPORTED;
        }
        // rescale from int8_t to int8_t
        if (param == ReduceSum || param == ReduceProd || param == ReduceMean) {
            status = PPLRescaleInt82Int8<float>(stream, output_shape, (const float*)tmp_buffer,
                (int8_t*)output, qparam);
        } else {
            status = PPLRescaleInt82Int8<int8_t>(stream, output_shape, (const int8_t*)tmp_buffer,
                (int8_t*)output, qparam);
        }
        return status;
    } else {
        return ppl::common::RC_UNSUPPORTED;
    }
    // return ppl::common::RC_SUCCESS;
#undef CASE
#undef CASEPROMOTION
}