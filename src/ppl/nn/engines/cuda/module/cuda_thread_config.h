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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_CUDA_THREAD_CONFIG_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_CUDA_THREAD_CONFIG_H_

namespace ppl { namespace nn { namespace cuda {

struct CUDAThreadConfig {
    // config array, first three are Grid dimension config
    // last three are Block dimension config
    size_t thread_config[6];
    // Dynamic shared memory size in bytes
    size_t dyn_shmem_size{0};
    /*
     *  param i The Block dimension
     *  return i-th block dim
     */
    size_t BlockDim(int i) {
        return thread_config[i + 3];
    }
    /*
     *  param i The Grid dimension
     *  return i-th grid dim
     */
    size_t GridDim(int i) {
        return thread_config[i];
    }
};

}}} // namespace ppl::nn::cuda

#endif