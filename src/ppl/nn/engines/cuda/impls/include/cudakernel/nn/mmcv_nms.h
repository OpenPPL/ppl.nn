#ifndef PPLCUDA_KERNEL_INCLUDE_MMCVNMS_MMCVNMS_H_
#define PPLCUDA_KERNEL_INCLUDE_MMCVNMS_MMCVNMS_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>
#include <float.h>

int64_t PPLMMCVNMSGetTempBufferSize(const ppl::nn::TensorShape* boxes_shape);

ppl::common::RetCode PPLCUDAMMCVNMSForwardImp(
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
    float iou_threshold,
    int64_t offset);

#endif //PPLCUDA_KERNEL_INCLUDE_MMCVNMS_MMCVNMS_H_
