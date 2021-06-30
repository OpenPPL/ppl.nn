#ifndef PPLCUDA_KERNEL_INCLUDE_MMCV_GRIDSAMPLE_GRIDSAMPLE_H_
#define PPLCUDA_KERNEL_INCLUDE_MMCV_GRIDSAMPLE_GRIDSAMPLE_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include "ppl/nn/params/mmcv/mmcv_gridsample_param.h"
#include <cuda_fp16.h>
#include <float.h>

ppl::common::RetCode PPLCUDAMMCVGridSampleForwardImp(
    cudaStream_t stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    ppl::nn::TensorShape* rois_shape,
    const void* rois,
    ppl::nn::TensorShape* output_shape,
    void* output,
    ppl::nn::common::MMCVGridSampleParam param);
#endif //PPLCUDA_KERNEL_INCLUDE_MMCV_GRIDSAMPLE_GRIDSAMPLE_H_
