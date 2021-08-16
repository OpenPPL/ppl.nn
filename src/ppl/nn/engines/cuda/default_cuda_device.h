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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_DEVICE_H_

#include "ppl/common/allocator.h"

#include "ppl/nn/engines/cuda/cuda_device.h"

namespace ppl { namespace nn { namespace cuda {

class DefaultCudaDevice final : public CudaDevice {
public:
    DefaultCudaDevice();
    virtual ~DefaultCudaDevice();

    ppl::common::RetCode Init(uint32_t device_id);
    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    void Free(BufferDesc*) override;

private:
    std::unique_ptr<ppl::common::Allocator> allocator_;
};

}}} // namespace ppl::nn::cuda

#endif
