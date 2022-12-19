// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for arithmeticitional information
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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_ARITHMETIC_NEON_ARITHMETIC_H_
#define __ST_PPL_KERNEL_ARM_SERVER_ARITHMETIC_NEON_ARITHMETIC_H_

#include "ppl/kernel/arm_server/common/general_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode add(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst);

ppl::common::RetCode sub(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst);

ppl::common::RetCode mul(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst);

ppl::common::RetCode div(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const void *src0,
    const void *src1,
    const bool fuse_relu,
    void *dst);

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_ARITHMETIC_NEON_ARITHMETIC_H_
