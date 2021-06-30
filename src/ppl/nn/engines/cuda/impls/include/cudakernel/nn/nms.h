#ifndef PPLCUDA_KERNEL_INCLUDE_NMS_NMS_H_
#define PPLCUDA_KERNEL_INCLUDE_NMS_NMS_H_
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_fp16.h>
#include <float.h>

int64_t PPLNMSGetTempBufferSize(const ppl::nn::TensorShape* scores_shape);

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
    float score_threshold = -FLT_MAX);

#endif //PPLCUDA_KERNEL_INCLUDE_NMS_NMS_H_
