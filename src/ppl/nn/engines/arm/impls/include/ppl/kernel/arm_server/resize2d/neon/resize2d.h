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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_RESIZE2D_NEON_RESIZE2D_H_
#define __ST_PPL_KERNEL_ARM_SERVER_RESIZE2D_NEON_RESIZE2D_H_

#include "ppl/kernel/arm_server/common/general_include.h"
#include "ppl/nn/params/onnx/resize_param.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode resize2d(
    const ppl::common::TensorShape *src_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src,
    const float scale_h,
    const float scale_w,
    const int32_t coord_trans_mode,
    const int32_t mode,
    const float cubic_coeff_a,
    void *dst);

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
