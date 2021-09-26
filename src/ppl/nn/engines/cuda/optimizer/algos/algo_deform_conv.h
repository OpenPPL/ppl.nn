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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_DEFORM_CONV_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_DEFORM_CONV_H_

#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"

#include "ppl/nn/params/mmcv/mmcv_modulated_deform_conv2d_param.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

class DeformConvAlgorithm : public Algorithm {
public:
    DeformConvAlgorithm() {
        std::set<dataformat_t> nhwc8{DATAFORMAT_NHWC8};
        conv_formats_.emplace(DATAFORMAT_NDARRAY, nhwc8);
    }

    const std::map<dataformat_t, std::set<dataformat_t>> Getformats(const std::string& type_name) const override {
        return conv_formats_;
    }

    void GetAttrParam(void*& param) const override {
        return;
    };
    void DeleteAttrParam(void*& param) override {
        return;
    };

public:
    double ExcuteTimer(const ir::Node* node, OptKernelOptions& options) override;
    RetCode ModifyParam(const ir::Node* node, OptKernelOptions& options) override;
    void ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                                ppl::common::dataformat_t input_format, ppl::common::dataformat_t output_format) override;

private:
    ppl::nn::common::MMCVModulatedDeformConv2dParam* param_ = nullptr;
    std::map<dataformat_t, std::set<dataformat_t>> conv_formats_;
};

}}} // namespace ppl::nn::cuda

#endif
