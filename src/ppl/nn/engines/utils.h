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

#ifndef _ST_HPC_PPL_NN_ENGINES_UTILS_H_
#define _ST_HPC_PPL_NN_ENGINES_UTILS_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/common/constant_visitor.h"
#include "ppl/nn/runtime/runtime_constant_info.h"
#include <map>

namespace ppl { namespace nn { namespace utils {

/** @brief copy buffer to tensor */
ppl::common::RetCode CopyBuffer(const BufferDesc& src_buf, const TensorShape& src_shape, const Device* src_device,
                                TensorImpl* dst, Device* tmp_cpu_device = nullptr);

/** @brief copy one tensor to another */
static inline ppl::common::RetCode CopyTensorBuffer(const TensorImpl& src, TensorImpl* dst,
                                                    Device* tmp_cpu_device = nullptr) {
    return CopyBuffer(src.GetBufferDesc(), *src.GetShape(), src.GetDevice(), dst, tmp_cpu_device);
}

ppl::common::RetCode LoadConstants(const ir::Graph&, Device*, std::map<edgeid_t, RuntimeConstantInfo>*,
                                   const std::set<edgeid_t>* = nullptr);

ppl::common::RetCode LoadConstants(const ConstantVisitor&, Device*, std::map<edgeid_t, BufferInfo>*);

ppl::common::RetCode GenericLoadConstant(const void* data, uint64_t size, const TensorShape& shape, Device* device,
                                         RuntimeConstantInfo* info, bool omit_data = false);

ppl::common::RetCode GenericLoadConstant(const void* data, uint64_t size, const TensorShape& shape, Device* device,
                                         BufferInfo* info);

void IrShape2TensorShape(const ir::Shape&, TensorShape*);

}}} // namespace ppl::nn::utils

#endif
