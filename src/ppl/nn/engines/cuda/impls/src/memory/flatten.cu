#include "cudakernel/memory/flatten.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/common/types.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAFlattenForwardImp(
    cudaStream_t stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* output_shape,
    void* output)
{
    int64_t num_elems_output = output_shape->GetElementsIncludingPadding();
    cudaMemcpyAsync(output, input, ppl::common::GetSizeOfDataType(input_shape->GetDataType()) * num_elems_output, cudaMemcpyDeviceToDevice, stream);
    return ppl::common::RC_SUCCESS;
}