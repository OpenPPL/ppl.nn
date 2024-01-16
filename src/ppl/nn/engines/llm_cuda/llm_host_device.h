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

#ifndef _ST_HPC_PPL_NN_LLM_HOST_DEVICE_H_
#define _ST_HPC_PPL_NN_LLM_HOST_DEVICE_H_

#include "ppl/nn/engines/llm_cuda/options.h"
#include "ppl/nn/common/device.h"
#include "ppl/nn/utils/compact_buffer_manager.h"
#include "ppl/common/allocator.h"

#include <memory>

namespace ppl { namespace nn { namespace llm { namespace cuda {

class LlmHostDevice final : public Device {
public:
    LlmHostDevice() {
        *(uint64_t*)(type_.str) = 0;
        type_.str[0] = 'c';
        type_.str[1] = 'p';
        type_.str[2] = 'u';
    };
    virtual ~LlmHostDevice();

    ppl::common::RetCode Init(uint32_t mm_policy);

    using Device::Realloc;
    ppl::common::RetCode Realloc(const TensorShape& shape, BufferDesc* buffer) override final {
        return Realloc(shape.CalcBytesIncludingPadding(), buffer);
    }
    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override final;

    using Device::Free;
    void Free(BufferDesc* buffer) override final;

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }
    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const override final;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape& shape) const override final;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const override final;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape& shape) const override final;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const override final;

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc,
                                       const void* src_custom_info = nullptr) override;
    ppl::common::RetCode ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                            const TensorShape& src_desc,
                                            const void* src_custom_info = nullptr) override;

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc,
                                         const void* dst_custom_info = nullptr) override;
    ppl::common::RetCode ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc,
                                         const void* dst_custom_info = nullptr) override;

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                 const TensorShape& src_desc, const void* dst_custom_info = nullptr,
                                 const void* src_custom_info = nullptr) override;

    const Type& GetType() const override final {
        return type_;
    }

    ppl::common::RetCode Synchronize() override final {
        return ppl::common::RC_SUCCESS;
    }

    // ----- configurations ----- //

    typedef ppl::common::RetCode (*ConfHandlerFunc)(LlmHostDevice*, va_list);
    static ConfHandlerFunc conf_handlers_[0];

    ppl::common::RetCode Configure(uint32_t, ...) override;

private:
    ppl::common::RetCode ConvertToHostCommon(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                             const TensorShape& src_desc, const void* src_custom_info,
                                             const std::function<ppl::common::RetCode(void*, const BufferDesc&, const TensorShape&)>& copy_fn);
    ppl::common::RetCode ConvertFromHostCommon(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                               const TensorShape& src_desc, const void* dst_custom_info,
                                               const std::function<ppl::common::RetCode(BufferDesc*, const void*, const TensorShape&)>& copy_fn);

private:
    Type type_;
    const uint64_t alignment_ = 64u;
    uint32_t mm_policy_;
    std::unique_ptr<utils::BufferManager> buffer_manager_;
    std::shared_ptr<ppl::common::CompactAddrManager::VMAllocator> vmr_;
    std::shared_ptr<ppl::common::Allocator> allocator_;

private:
    LlmHostDevice(const LlmHostDevice&) = delete;
    LlmHostDevice& operator=(const LlmHostDevice&) = delete;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
