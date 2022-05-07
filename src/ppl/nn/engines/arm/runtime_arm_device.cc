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

#include "ppl/nn/engines/arm/runtime_arm_device.h"
#include "ppl/nn/engines/arm/options.h"
#include "ppl/nn/utils/stack_buffer_manager.h"
#include "ppl/nn/utils/compact_buffer_manager.h"
#include "ppl/nn/utils/cpu_block_allocator.h"
#include "ppl/nn/common/logger.h"
#include <stdarg.h>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

static void DummyDeleter(ppl::common::Allocator*) {}

RuntimeArmDevice::RuntimeArmDevice(uint64_t alignment, isa_t isa, uint32_t mm_policy)
    : ArmDevice(alignment, isa), tmp_buffer_size_(0) {
    can_defragement_ = false;
    if (mm_policy == MM_MRU) {
        auto allocator_ptr = ArmDevice::GetAllocator();
        allocator_ = std::shared_ptr<Allocator>(allocator_ptr, DummyDeleter);
        buffer_manager_.reset(new utils::StackBufferManager(allocator_ptr));
    } else if (mm_policy == MM_COMPACT) {
        can_defragement_ = true;
        allocator_.reset(new utils::CpuBlockAllocator());
        buffer_manager_.reset(new utils::CompactBufferManager(allocator_.get(), alignment, 64u));
    }
}

RuntimeArmDevice::~RuntimeArmDevice() {
    LOG(DEBUG) << "buffer manager[" << buffer_manager_->GetName() << "] allocates ["
               << buffer_manager_->GetAllocatedBytes() << "] bytes.";
    if (tmp_buffer_size_) {
        buffer_manager_->Free(&shared_tmp_buffer_);
    }
    buffer_manager_.reset();
}

RetCode RuntimeArmDevice::AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
    if (can_defragement_) {
        auto ret = buffer_manager_->Realloc(bytes, &shared_tmp_buffer_);
        if (RC_SUCCESS != ret) {
            return ret;
        }
    } else {
        if (bytes > tmp_buffer_size_ || bytes <= tmp_buffer_size_ / 2) {
            auto ret = buffer_manager_->Realloc(bytes, &shared_tmp_buffer_);
            if (RC_SUCCESS != ret) {
                return ret;
            }
            tmp_buffer_size_ = bytes;
        }
    }
    *buffer = shared_tmp_buffer_;
    return RC_SUCCESS;
}

void RuntimeArmDevice::FreeTmpBuffer(BufferDesc* buffer) {
    if (can_defragement_) {
        buffer_manager_->Free(&shared_tmp_buffer_);
    }
}

RetCode RuntimeArmDevice::DoMemDefrag(RuntimeArmDevice* dev, va_list) {
    if (!dev->can_defragement_) {
        return RC_UNSUPPORTED;
    }

    auto mgr = dynamic_cast<utils::CompactBufferManager*>(dev->buffer_manager_.get());
    if (dev->tmp_buffer_size_ > 0) {
        mgr->Free(&dev->shared_tmp_buffer_);
        dev->shared_tmp_buffer_.addr = nullptr;
        dev->tmp_buffer_size_ = 0;
    }
    return mgr->Defragment();
}

RuntimeArmDevice::ConfHandlerFunc RuntimeArmDevice::conf_handlers_[] = {
    DoMemDefrag, // DEV_CONF_MEM_DEFRAG
};

RetCode RuntimeArmDevice::Configure(uint32_t option, ...) {
    if (option >= DEV_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << DEV_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}} // namespace ppl::nn::arm
