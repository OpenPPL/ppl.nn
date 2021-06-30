#include "cudakernel/nn/softmax.h"
#include "cudakernel/common/divmod_fast.h"
#include "cudakernel/reduce/reduce.h"
#include "cudakernel/arithmetic/arithmetic.h"
#include "cudakernel/unary/exp.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>
#include <float.h>
#include <memory>

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
    void *max_sum_output   = temp_buffer;
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
    status = PPLCUDAArithMeticSubForwardImp(stream, &nd_shape, input, &max_sum_shape, max_sum_output, &nd_shape, output);
    // exp
    status = PPLCUDAExpForwardImp(stream, &nd_shape, output, &nd_shape, output);
    //reduce sum
    ReduceParam reduce_sum = ReduceSum;
    status = PPLCUDAReduceForwardImp(stream, reduce_sum, reduce_desc, &nd_shape, output, &max_sum_shape, max_sum_output);
    //div
    status = PPLCUDAArithMeticDivForwardImp(stream, &nd_shape, output, &max_sum_shape, max_sum_output, &nd_shape, output);
    return status;
}
