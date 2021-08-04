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

#ifndef _ST_HPC_PPL_NN_PYTHON_PY_CUDA_ENGINE_H_
#define _ST_HPC_PPL_NN_PYTHON_PY_CUDA_ENGINE_H_

#include "ppl/nn/engines/engine.h"
#include "pybind11/pybind11.h"

namespace ppl { namespace nn { namespace python {

class PyCudaEngine final {
public:
    PyCudaEngine(Engine* engine) : engine_(engine) {}
    PyCudaEngine(PyCudaEngine&&) = default;
    PyCudaEngine& operator=(PyCudaEngine&&) = default;
    PyCudaEngine(const PyCudaEngine&) = default;
    PyCudaEngine& operator=(const PyCudaEngine&) = default;

    std::string GetName() const {
        return engine_->GetName();
    }
    const std::shared_ptr<Engine>& GetInnerPtr() const {
        return engine_;
    }
    ppl::common::RetCode Configure(uint32_t option, const pybind11::args& args);

private:
    std::shared_ptr<Engine> engine_;
};

}}} // namespace ppl::nn::python

#endif
