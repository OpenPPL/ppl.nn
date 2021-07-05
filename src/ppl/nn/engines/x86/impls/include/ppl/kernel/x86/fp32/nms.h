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

#ifndef __ST_PPL_KERNEL_X86_FP32_NMS_H_
#define __ST_PPL_KERNEL_X86_FP32_NMS_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode nms_ndarray_fp32(
    const float *boxes,
    const float *scommons,
    const uint32_t num_boxes_in,
    const uint32_t batch,
    const uint32_t num_classes,
    const bool center_point_box,
    const int64_t max_output_boxes_per_batch_per_class,
    const float iou_threshold,
    const float scommon_threshold,
    int64_t *dst,
    int64_t *num_boxes_out);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_NMS_H_
