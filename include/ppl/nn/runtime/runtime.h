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

#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/common.h"
#include "ppl/nn/common/device_context.h"
#include "ppl/nn/runtime/tensor.h"
#include "ppl/nn/runtime/profiling_statistics.h"

namespace ppl { namespace nn {

/** options for Runtime::Configure() */
enum {
    /**
       @brief args: true/false.
       @note this option may cause performance loss
    */
    RUNTIME_CONF_SET_KERNEL_PROFILING_FLAG = 0,

    RUNTIME_CONF_MAX,
};

/**
   @class Runtime
   @brief runs a model
*/
class PPLNN_PUBLIC Runtime {
public:
    virtual ~Runtime() {}

    /**
       @brief set various runtime options defined in `runtime_options.h`.
       parameters vary depending on the first parameter `option`.
    */
    virtual ppl::common::RetCode Configure(uint32_t option, ...) = 0;

    /** @brief get the number of inputs of the associated graph. */
    virtual uint32_t GetInputCount() const = 0;

    /**
       @brief get input tensor at position `idx`.
       @param idx should be less than `GetInputCount()`.
    */
    virtual Tensor* GetInputTensor(uint32_t idx) const = 0;

    /**
       @brief run the model with given inputs.
       @note input data must be filled via the returned value of `GetInputTensor()`
       before calling this function.
    */
    virtual ppl::common::RetCode Run() = 0;

    /**
       @brief blocks until all operations finish.
       @note MUST be called before getting outputs or profiling statistics in case
       some engine may run asynchronously.
    */
    virtual ppl::common::RetCode Sync() = 0;

    /** @brief get the number of outputs of the associated graph. */
    virtual uint32_t GetOutputCount() const = 0;

    /**
       @brief get output tensor at position `idx`.
       @param idx should be less than `GetOutputCount()`.
    */
    virtual Tensor* GetOutputTensor(uint32_t idx) const = 0;

    /** @brief get the number of `DeviceContext` used by this `Runtime` instance */
    virtual uint32_t GetDeviceContextCount() const = 0;

    /**
       @brief get device context at position `idx`.
       @param idx should be less than `GetDeviceContextCount()`.
    */
    virtual DeviceContext* GetDeviceContext(uint32_t idx) const = 0;

    /**
       @brief get profiling statistics of each kernel.
       @note alailable if `PPLNN_ENABLE_KERNEL_PROFILING` is enabled.
    */
    virtual ppl::common::RetCode GetProfilingStatistics(ProfilingStatistics*) const = 0;
};

}} // namespace ppl::nn

#endif
