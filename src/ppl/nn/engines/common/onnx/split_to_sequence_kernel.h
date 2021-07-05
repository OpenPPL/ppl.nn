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

#ifndef _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SPLIT_TO_SEQUENCE_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_COMMON_ONNX_SPLIT_TO_SEQUENCE_KERNEL_H_

#include "ppl/nn/engines/common/common_kernel_impl.h"
#include "ppl/nn/params/onnx/split_to_sequence_param.h"

namespace ppl { namespace nn { namespace common {

class SplitToSequenceKernel final : public CommonKernelImpl {
public:
    typedef std::function<ppl::common::RetCode(uint64_t dims_before_axis, uint64_t dims_from_axis,
                                               uint64_t dims_after_axis, uint32_t dims_of_chunk, uint32_t element_size,
                                               Device* device,
                                               BufferDesc* src_cursor, // in && out, can be modified for next round
                                               BufferDesc* dst)>
        SplitFunc;

public:
    SplitToSequenceKernel(const ir::Node* node) : CommonKernelImpl(node) {}
    void SetExecutionInfo(uint64_t axis, uint64_t keepdims, const SplitFunc& f);

protected:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

private:
    uint64_t axis_, keepdims_;
    SplitFunc split_func_;
};

}}} // namespace ppl::nn::common

#endif
