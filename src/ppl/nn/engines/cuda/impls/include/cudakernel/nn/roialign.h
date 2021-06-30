#ifndef PPLCUDA_KERNEL_INCLUDE_ROIALIGN_ROIALIGN_H_
#define PPLCUDA_KERNEL_INCLUDE_ROIALIGN_ROIALIGN_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/params/onnx/roialign_param.h"
#include <cuda_fp16.h>
#include <float.h>

ppl::common::RetCode PPLCUDAROIAlignForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* rois_shape,
    const void* rois,
    ppl::nn::TensorShape* batch_indices_shape,
    const void* batch_indices,
    ppl::nn::TensorShape* output_shape,
    void* output,
    ppl::nn::common::ROIAlignParam param);
#endif //PPLCUDA_KERNEL_INCLUDE_ONNX_ROIALIGN_ROIALIGN_H_
