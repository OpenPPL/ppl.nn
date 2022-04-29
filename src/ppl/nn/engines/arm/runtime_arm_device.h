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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_RUNTIME_ARM_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_RUNTIME_ARM_DEVICE_H_

#include "ppl/nn/engines/arm/arm_device.h"
#include "ppl/nn/engines/arm/arm_options.h"
#include "ppl/nn/utils/buffer_manager.h"
#include "ppl/common/allocator.h"
#include <memory>

namespace ppl { namespace nn { namespace arm {

class RuntimeArmDevice final : public ArmDevice {
private:
    static inline uint64_t Align(uint64_t x, uint64_t n) {
        return (x + n - 1) & (~(n - 1));
    }

public:
    RuntimeArmDevice(uint64_t alignment, ppl::common::isa_t isa, uint32_t mm_policy);
    ~RuntimeArmDevice();

    ppl::common::Allocator* GetAllocator() const override {
        return allocator_.get();
    }

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override {
        return buffer_manager_->Realloc(bytes, buffer);
    }

    void Free(BufferDesc* buffer) override {
        buffer_manager_->Free(buffer);
    }

    ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) override;
    void FreeTmpBuffer(BufferDesc* buffer) override;

    // ----- configurations ----- //

    /**
       @brief replaces all blocks with a single block.
       @note make sure that this device is not used when calling DoMemDefrag().
    */
    static ppl::common::RetCode DoMemDefrag(RuntimeArmDevice*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(RuntimeArmDevice*, va_list);
    static ConfHandlerFunc conf_handlers_[ARM_DEV_CONF_MAX];

    ppl::common::RetCode Configure(uint32_t, ...) override;

private:
    bool can_defragement_;
    BufferDesc shared_tmp_buffer_;
    uint64_t tmp_buffer_size_;
    std::unique_ptr<utils::BufferManager> buffer_manager_;
    std::shared_ptr<ppl::common::Allocator> allocator_;
};

}}} // namespace ppl::nn::arm

#endif
