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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_BARRIER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_BARRIER_H_

#include "ppl/nn/runtime/barrier.h"
#include <cuda_runtime.h>

namespace ppl { namespace nn {

class CudaBarrier final : public Barrier {
public:
    ~CudaBarrier() {
        cudaEventDestroy(event_);
    }

    ppl::common::RetCode Init() {
        auto err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaEventCreateWithFlags failed: " << cudaGetErrorString(err);
            return ppl::common::RC_OTHER_ERROR;
        }

        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode Update(cudaStream_t stream) {
        auto err = cudaEventRecord(event_, stream);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaEventRecord failed: " << cudaGetErrorString(err);
            return ppl::common::RC_OTHER_ERROR;
        }
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode Sync() override {
        auto err = cudaEventSynchronize(event_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaEventSynchronize failed: " << cudaGetErrorString(err);
            return ppl::common::RC_OTHER_ERROR;
        }
        return ppl::common::RC_SUCCESS;
    }

private:
    cudaEvent_t event_;
};

}} // namespace ppl::nn

#endif
