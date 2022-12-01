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

#ifndef __ST_PPL_KERNEL_RISCV_FP16_RESIZE_H_
#define __ST_PPL_KERNEL_RISCV_FP16_RESIZE_H_

#include "ppl/kernel/riscv/common/general_include.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode channel_shuffle_ndarray_fp16(const ppl::common::TensorShape* src_shape, const __fp16* src, const int32_t group, __fp16* dst);

// 2nd output is optional, do not fuse split while it is empty
ppl::common::RetCode channel_shuffle_ndarray_concat_split_fp16(const ppl::common::TensorShape* src0_shape,
                                                               const ppl::common::TensorShape* src1_shape,
                                                               const __fp16* src0,
                                                               const __fp16* src1,
                                                               const int32_t group,
                                                               __fp16* dst0,
                                                               __fp16* dst1_optional);

ppl::common::RetCode channel_shuffle_n8cx_fp16(const ppl::common::TensorShape* src_shape, const __fp16* src, const int32_t group, __fp16* dst);

// 2nd output is optional, do not fuse split while it is empty
ppl::common::RetCode channel_shuffle_n8cx_concat_split_fp16(const ppl::common::TensorShape* src0_shape,
                                                            const ppl::common::TensorShape* src1_shape,
                                                            const __fp16* src0,
                                                            const __fp16* src1,
                                                            const int32_t group,
                                                            __fp16* dst0,
                                                            __fp16* dst1_optional);

}}}; // namespace ppl::kernel::riscv

#endif
