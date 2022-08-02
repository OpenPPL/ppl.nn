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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_ARM_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_ARM_DEVICE_H_

#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/nn/common/device.h"
#include "ppl/nn/engines/arm/data_converter.h"
#include <cstring>

namespace ppl { namespace nn { namespace arm {

class ArmDevice : public Device {
public:
    ArmDevice(uint64_t alignment, ppl::common::isa_t isa) : isa_(isa), allocator_(alignment), data_converter_(isa) {}

    void SetISA(ppl::common::isa_t isa) {
        isa_ = isa;
        data_converter_.SetISA(isa);
    }
    const ppl::common::isa_t GetISA() const {
        return isa_;
    }

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }

    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    virtual ppl::common::Allocator* GetAllocator() const {
        return &allocator_;
    }

    const DataConverter* GetDataConverter() const override final {
        return &data_converter_;
    }

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override {
        if (buffer->addr) {
            allocator_.Free(buffer->addr);
        }

        if (bytes == 0) {
            buffer->addr = nullptr;
            return ppl::common::RC_SUCCESS;
        }

        buffer->addr = allocator_.Alloc(bytes);
        if (!buffer->addr) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        buffer->desc = bytes;
        return ppl::common::RC_SUCCESS;
    }

    void Free(BufferDesc* buffer) override {
        if (buffer->addr) {
            allocator_.Free(buffer->addr);
            buffer->addr = nullptr;
        }
    }

    ppl::common::RetCode Realloc(const TensorShape& shape, BufferDesc* buffer) override final {
        return Realloc(shape.CalcBytesIncludingPadding(), buffer);
    }

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override final {
        memcpy(dst->addr, src, bytes);
        return ppl::common::RC_SUCCESS;
    }
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const override final {
        return CopyFromHost(dst, src, shape.CalcBytesIncludingPadding());
    }

    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override final {
        memcpy(dst, src.addr, bytes);
        return ppl::common::RC_SUCCESS;
    }
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const override final {
        return CopyToHost(dst, src, shape.CalcBytesIncludingPadding());
    }

    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override final {
        memcpy(dst->addr, src.addr, bytes);
        return ppl::common::RC_SUCCESS;
    }
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const override final {
        return Copy(dst, src, shape.CalcBytesIncludingPadding());
    }

    ppl::common::RetCode Sync() override final {
        return ppl::common::RC_SUCCESS;
    }

    const char* GetType() const override final {
        return "arm";
    }

    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }

private:
    ppl::common::isa_t isa_;
    mutable ppl::common::GenericCpuAllocator allocator_;
    ArmDataConverter data_converter_;
};

}}} // namespace ppl::nn::arm

#endif
