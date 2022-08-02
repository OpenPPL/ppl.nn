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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_X86_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_X86_DEVICE_H_

#include "ppl/nn/common/device.h"
#include "ppl/nn/engines/x86/data_converter.h"
#include "ppl/common/generic_cpu_allocator.h"
#include <cstring> // memcpy

namespace ppl { namespace nn { namespace x86 {

class X86Device : public Device {
public:
    X86Device(uint64_t alignment, ppl::common::isa_t isa) : isa_(isa), data_converter_(isa), allocator_(alignment) {}

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

    const DataConverter* GetDataConverter() const override final {
        return &data_converter_;
    }

    const char* GetType() const override final {
        return "x86";
    }

    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }

private:
    ppl::common::isa_t isa_;
    X86DataConverter data_converter_;
    mutable ppl::common::GenericCpuAllocator allocator_;
};

}}} // namespace ppl::nn::x86

#endif
