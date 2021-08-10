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

#ifndef _ST_HPC_PPL_NN_RUNTIME_TENSOR_H_
#define _ST_HPC_PPL_NN_RUNTIME_TENSOR_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/common/common.h"

namespace ppl { namespace nn {

class PPLNN_PUBLIC Tensor {
public:
    virtual ~Tensor() {}

    /** @brief get tensor's name */
    virtual const char* GetName() const = 0;

    /** @brief get tensor's shape */
    virtual TensorShape& GetShape() = 0;

    /** @brief get tensor's shape */
    virtual const TensorShape& GetShape() const = 0;

    /** @brief rellocate a buffer according to its shape */
    virtual ppl::common::RetCode ReallocBuffer() = 0;

    /**
       @brief copy tensor's data to `dst`, which points to a host memory
       @note `dst` MUST have enough space.
    */
    virtual ppl::common::RetCode CopyToHost(void* dst) const = 0;

    /**
       @brief copy tensor's data from `dst`, which points to a host memory
       @note `dst` MUST have enough space.
    */
    virtual ppl::common::RetCode CopyFromHost(const void* src) = 0;

    /** @brief convert tensor's data to `dst` with shape `dst_desc` */
    virtual ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc) const = 0;

    /** @brief convert tensor's data from `dst` with shape `dst_desc` */
    virtual ppl::common::RetCode ConvertFromHost(const void* src, const TensorShape& src_desc) = 0;

    /**
       @brief set the underlying buffer ptr
       @param buf buffer ptr that can be read/written by the internal `Device` class.
    */
    virtual void SetBufferPtr(void* buf) = 0;

    /** @brief get the underlying buffer ptr */
    virtual void* GetBufferPtr() const = 0;
};

}} // namespace ppl::nn

#endif
