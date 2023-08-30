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
#include "ppl/common/generic_cpu_allocator.h"

namespace ppl { namespace nn { namespace utils {

class GenericCpuDevice final : public Device {
public:
    GenericCpuDevice(uint64_t alignment = 64) : allocator_(alignment) {
        *(uint64_t*)(type_.str) = 0;
        type_.str[0] = 'c';
        type_.str[1] = 'p';
        type_.str[2] = 'u';
    }

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    ppl::common::RetCode Realloc(const TensorShape&, BufferDesc*) override final;
    void Free(BufferDesc*) override;

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const override;
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape&) const override;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape&) const override;

    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const override;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape&) const override;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape&) const override;

    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape&) const override;

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc, const void* src_custom_info = nullptr) override;
    ppl::common::RetCode ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                            const TensorShape& src_desc,
                                            const void* src_custom_info = nullptr) override;
    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc, const void* dst_custom_info = nullptr) override;
    ppl::common::RetCode ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                              const TensorShape& src_desc,
                                              const void* dst_custom_info = nullptr) override;
    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                 const TensorShape& src_desc, const void* dst_custom_info = nullptr,
                                 const void* src_custom_info = nullptr) override;

    const Type& GetType() const override {
        return type_;
    }

    ppl::common::RetCode Synchronize() override {
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }

private:
    Type type_;
    mutable ppl::common::GenericCpuAllocator allocator_;
};

}}} // namespace ppl::nn::utils

#endif
