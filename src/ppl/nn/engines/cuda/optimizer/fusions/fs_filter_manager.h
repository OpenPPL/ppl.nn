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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_FS_FILTER_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_FS_FILTER_MANAGER_H_

#include "ppl/nn/engines/cuda/optimizer/fusions/fusion.h"

#include "ppl/nn/engines/cuda/optimizer/fusions/fs_averagepool.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_channel_shuffle.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_concat.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_conv.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_gemm.h"
#include "ppl/nn/engines/cuda/optimizer/fusions/fs_softmax.h"

namespace ppl { namespace nn { namespace cuda {

class FsFilterManager {
public:
    static FsFilterManager* Instance() {
        static FsFilterManager mgr;
        return &mgr;
    }

    Fusion* FindFusion(const std::string& kernel_type) const;

private:
    FsFilterManager();

private:
    std::map<std::string, Fusion*> type2fusion_;
    AveragePoolFusion averagepool_fs_;
    ConcatFusion concat_fs_;
    ChannelShuffleFusion channel_shuffle_fs_;
    ConvFusion conv_fs_;
    GemmFusion gemm_fs_;
    SoftmaxFusion softmax_fs_;
};

}}} // namespace ppl::nn::cuda

#endif
