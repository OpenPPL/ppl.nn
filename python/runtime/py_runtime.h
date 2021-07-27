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

#ifndef _ST_HPC_PPL_NN_PYTHON_PY_RUNTIME_H_
#define _ST_HPC_PPL_NN_PYTHON_PY_RUNTIME_H_

#include "../engines/py_engine.h"
#include "ppl/nn/runtime/runtime.h"
#include "py_tensor.h"
#include <memory>

namespace ppl { namespace nn { namespace python {

class PyRuntime final {
public:
    PyRuntime(const std::vector<PyEngine>& engines, Runtime* runtime) : engines_(engines), runtime_(runtime) {}
    PyRuntime(PyRuntime&&) = default;
    PyRuntime& operator=(PyRuntime&&) = default;

    uint32_t GetInputCount() const {
        return runtime_->GetInputCount();
    }
    PyTensor GetInputTensor(uint32_t idx) const {
        return PyTensor(runtime_->GetInputTensor(idx));
    }
    ppl::common::RetCode Run() {
        return runtime_->Run();
    }
    ppl::common::RetCode Sync() {
        return runtime_->Sync();
    }
    uint32_t GetOutputCount() const {
        return runtime_->GetOutputCount();
    }
    PyTensor GetOutputTensor(uint32_t idx) const {
        return PyTensor(runtime_->GetOutputTensor(idx));
    }

private:
    std::vector<PyEngine> engines_; // retain engines
    std::unique_ptr<Runtime> runtime_;
};

}}} // namespace ppl::nn::python

#endif
