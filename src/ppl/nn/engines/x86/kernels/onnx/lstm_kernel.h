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

#ifndef _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_LSTM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_KERNELS_ONNX_LSTM_KERNEL_H_

#include "ppl/nn/params/onnx/lstm_param.h"
#include "ppl/nn/engines/x86/kernel.h"
#include "ppl/kernel/x86/fp32/lstm.h"

namespace ppl { namespace nn { namespace x86 {

class LSTMKernel : public X86Kernel {
public:
    LSTMKernel(const ir::Node* node) : X86Kernel(node) {}
    bool CanDoExecute(const KernelExecContext& ctx) const;

    void SetParam(const ppl::nn::common::LSTMParam* p) {
        param_ = p;
        if (p->direction == ppl::nn::common::LSTMParam::DIR_FORWARD) {
            direction_ = ppl::kernel::x86::rnn_direction::FORWARD;
        }
        if (p->direction == ppl::nn::common::LSTMParam::DIR_REVERSE) {
            direction_ = ppl::kernel::x86::rnn_direction::REVERSE;
        }
        if (p->direction == ppl::nn::common::LSTMParam::DIR_BIDIRECTIONAL) {
            direction_ = ppl::kernel::x86::rnn_direction::BIDIRECTIONAL;
        }
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;

    const ppl::nn::common::LSTMParam* param_ = nullptr;
    ppl::kernel::x86::rnn_direction_t direction_;
};

}}} // namespace ppl::nn::x86

#endif
