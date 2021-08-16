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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_BUFFERED_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_BUFFERED_CUDA_DEVICE_H_

#include <cuda.h>

#include "ppl/nn/utils/buffer_manager.h"
#include "ppl/nn/engines/cuda/cuda_device.h"

namespace ppl { namespace nn { namespace cuda {

class BufferedCudaDevice final : public CudaDevice {
public:
    ~BufferedCudaDevice();

    ppl::common::RetCode Init(const CudaEngineOptions& options);

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    void Free(BufferDesc*) override;

    ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) override;
    void FreeTmpBuffer(BufferDesc* buffer) override;

private:
    std::unique_ptr<ppl::common::Allocator> allocator_;
    std::unique_ptr<utils::BufferManager> buffer_manager_;
    BufferDesc shared_tmp_buffer_;
    uint64_t tmp_buffer_size_ = 0;
};

}}} // namespace ppl::nn::cuda

#endif
