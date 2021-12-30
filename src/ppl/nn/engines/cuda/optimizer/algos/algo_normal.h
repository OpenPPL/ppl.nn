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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_NORMAL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_ALGOS_ALGO_NORMAL_H_

#include "ppl/nn/engines/cuda/optimizer/algos/algorithm.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

class NormalAlgorithm : public Algorithm {
public:
    NormalAlgorithm() {
        std::set<dataformat_t> ndarray{DATAFORMAT_NDARRAY};
        std::set<dataformat_t> nhwc8{DATAFORMAT_NHWC8};
        std::set<dataformat_t> nhwc16{DATAFORMAT_NHWC16};

        ndarray_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        inherited_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        inherited_formats_.emplace(DATAFORMAT_NHWC8, nhwc8);
        inherited_formats_.emplace(DATAFORMAT_NHWC16, nhwc16);
        arbitrary_formats_.emplace(DATAFORMAT_NDARRAY, ndarray);
        arbitrary_formats_.emplace(DATAFORMAT_NHWC8, ndarray);
        arbitrary_formats_.emplace(DATAFORMAT_NHWC16, ndarray);
    }

    void GetAttrParam(void*& param) const override {
        return;
    };
    void DeleteAttrParam(void*& param) override {
        return;
    };
    const std::map<dataformat_t, std::set<dataformat_t>> Getformats(const std::string& type_name) const override {
        if (inherited_set_.find(type_name) != inherited_set_.end()) {
            return inherited_formats_;
        }
        if (arbitrary_set_.find(type_name) != arbitrary_set_.end()) {
            return arbitrary_formats_;
        }
        return ndarray_formats_;
    }

    double ExcuteTimer(const ir::Node* node, OptKernelOptions& options) override;
    RetCode ModifyParam(ir::Node* node, OptKernelOptions& options) override {
        return RC_SUCCESS;
    }
    void ReshapeOnEdges(const ir::Node* node, std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors,
                        ppl::common::dataformat_t input_format, ppl::common::dataformat_t output_format) override;

private:
    std::map<dataformat_t, std::set<dataformat_t>> ndarray_formats_;
    std::map<dataformat_t, std::set<dataformat_t>> nhwc8_formats_;
    std::map<dataformat_t, std::set<dataformat_t>> inherited_formats_;
    std::map<dataformat_t, std::set<dataformat_t>> arbitrary_formats_;
    std::set<std::string> inherited_set_{"Add",
                                         "Mul",
                                         "Div",
                                         "Sub",
                                         "Relu",
                                         "PRelu",
                                         "Clip",
                                         "LeakyRelu",
                                         "Cast",
                                         "MaxPool",
                                         "GlobalMaxPool",
                                         "AveragePool",
                                         "GlobalAveragePool",
                                         "BatchNormalization",
                                         "Slice",
                                         "Split",
                                         "Sigmoid",
                                         "ChannelShuffle"};
    std::set<std::string> arbitrary_set_{"Shape"};
};

}}} // namespace ppl::nn::cuda

#endif
