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

#ifndef _ST_HPC_PPL_NN_UTILS_GENERIC_CPU_DEVICE_H_
#define _ST_HPC_PPL_NN_UTILS_GENERIC_CPU_DEVICE_H_

#include "ppl/nn/common/device.h"
#include "ppl/nn/utils/generic_cpu_data_converter.h"
#include "ppl/common/generic_cpu_allocator.h"

namespace ppl { namespace nn { namespace utils {

class GenericCpuDevice : public Device {
public:
    GenericCpuDevice(uint64_t alignment = 64) : allocator_(alignment) {}
    virtual ~GenericCpuDevice() {}

    /** @brief get the underlying allocator used to allocate/free memories */
    ppl::common::Allocator* GetAllocator() const {
        return &allocator_;
    }

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    ppl::common::RetCode Realloc(const TensorShape&, BufferDesc*) override final;
    void Free(BufferDesc*) override;

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override;
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape&) const override;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape&) const override;

    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape&) const override;

    const DataConverter* GetDataConverter() const override {
        return &data_converter_;
    }

    DeviceContext* GetContext() const override {
        return &context_;
    }

private:
    class CpuDeviceContext final : public DeviceContext {
    public:
        const char* GetType() const override {
            return "cpu";
        }
        ppl::common::RetCode Configure(uint32_t, ...) override {
            return ppl::common::RC_UNSUPPORTED;
        }
    };

private:
    mutable ppl::common::GenericCpuAllocator allocator_;
    mutable CpuDeviceContext context_;
    GenericCpuDataConverter data_converter_;
};

}}} // namespace ppl::nn::utils

#endif
