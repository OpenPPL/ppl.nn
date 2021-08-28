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

#ifndef _ST_HPC_PPL_NN_PYTHON_PY_RUNTIME_BUILDER_H_
#define _ST_HPC_PPL_NN_PYTHON_PY_RUNTIME_BUILDER_H_

#include "../engines/py_engine.h"
#include "py_runtime.h"
#include "ppl/nn/runtime/runtime_builder.h"
#include <memory>

namespace ppl { namespace nn { namespace python {

class PyRuntimeBuilder final {
public:
    PyRuntimeBuilder(const std::vector<PyEngine>& engines, RuntimeBuilder* builder)
        : engines_(engines), builder_(builder) {}
    PyRuntimeBuilder(PyRuntimeBuilder&&) = default;
    PyRuntimeBuilder& operator=(PyRuntimeBuilder&&) = default;

    PyRuntime CreateRuntime() {
        return PyRuntime(engines_, builder_->CreateRuntime());
    }
    RuntimeBuilder* GetPtr() const {
        return builder_.get();
    }

private:
    std::vector<PyEngine> engines_; // retain engines
    std::unique_ptr<RuntimeBuilder> builder_;
};

}}} // namespace ppl::nn::python

#endif
