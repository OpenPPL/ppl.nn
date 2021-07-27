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

#ifndef _ST_HPC_PPL_NN_PYTHON_PY_X86_ENGINE_H_
#define _ST_HPC_PPL_NN_PYTHON_PY_X86_ENGINE_H_

#include "ppl/nn/engines/engine.h"
#include "pybind11/pybind11.h"

namespace ppl { namespace nn { namespace python {

class PyX86Engine final {
public:
    PyX86Engine(Engine* engine) : engine_(engine) {}
    PyX86Engine(PyX86Engine&&) = default;
    PyX86Engine& operator=(PyX86Engine&&) = default;
    PyX86Engine(const PyX86Engine&) = default;
    PyX86Engine& operator=(const PyX86Engine&) = default;

    std::string GetName() const {
        return engine_->GetName();
    }
    const std::shared_ptr<Engine>& GetEnginePtr() const {
        return engine_;
    }
    ppl::common::RetCode Configure(uint32_t option, const pybind11::args& args);

private:
    std::shared_ptr<Engine> engine_;
};


}}}

#endif
