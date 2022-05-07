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

#include "pybind11/pybind11.h"
#include "py_type_creator_manager.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

#ifdef PPLNN_USE_CUDA
void RegisterCudaBuiltinOpImpls();
void RegisterCudaEngineFactory(pybind11::module*);
void RegisterCudaEngineOptions(pybind11::module*);
void RegisterCudaEngine(pybind11::module*);
#endif

#ifdef PPLNN_USE_X86
void RegisterX86BuiltinOpImpls();
void RegisterX86EngineFactory(pybind11::module*);
void RegisterX86EngineOptions(pybind11::module*);
void RegisterX86Engine(pybind11::module*);
#endif

#ifdef PPLNN_USE_RISCV
void RegisterRiscvBuiltinOpImpls();
void RegisterRiscvEngineFactory(pybind11::module*);
void RegisterRiscvEngineOptions(pybind11::module*);
void RegisterRiscvEngine(pybind11::module*);
#endif

#ifdef PPLNN_USE_ARM
void RegisterArmBuiltinOpImpls();
void RegisterArmEngineFactory(pybind11::module*);
void RegisterArmEngineOptions(pybind11::module*);
void RegisterArmEngine(pybind11::module*);
#endif

void RegisterTensorShape(pybind11::module*);
void RegisterTensor(pybind11::module*);
void RegisterNdArray(pybind11::module*);
void RegisterEngine(pybind11::module*);
void RegisterDeviceContext(pybind11::module*);
void RegisterRuntime(pybind11::module*);
void RegisterVersion(pybind11::module*);

#ifdef PPLNN_ENABLE_ONNX_MODEL
void RegisterOnnxRuntimeBuilder(pybind11::module*);
void RegisterOnnxRuntimeBuilderFactory(pybind11::module*);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
void RegisterPmxRuntimeBuilder(pybind11::module*);
void RegisterPmxRuntimeBuilderFactory(pybind11::module*);
#endif

PYBIND11_MODULE(nn, m) {
    RegisterTensorShape(&m);
    RegisterTensor(&m);
    RegisterNdArray(&m);
    RegisterEngine(&m);
    RegisterDeviceContext(&m);
    RegisterRuntime(&m);
    RegisterVersion(&m);

#ifdef PPLNN_ENABLE_ONNX_MODEL
    pybind11::module onnx_module = m.def_submodule("onnx");
    RegisterOnnxRuntimeBuilderFactory(&onnx_module);
    RegisterOnnxRuntimeBuilder(&onnx_module);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
    pybind11::module pmx_module = m.def_submodule("pmx");
    RegisterPmxRuntimeBuilderFactory(&pmx_module);
    RegisterPmxRuntimeBuilder(&pmx_module);
#endif

    auto mgr = PyTypeCreatorManager::Instance();
    for (uint32_t i = 0; i < mgr->GetCreatorCount(); ++i) {
        auto creator = mgr->GetCreator(i);
        auto status = creator->Register(&m);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "register python type failed.";
            return;
        }
    }

#ifdef PPLNN_USE_CUDA
    pybind11::module cuda_module = m.def_submodule("cuda");
    RegisterCudaBuiltinOpImpls();
    RegisterCudaEngineFactory(&cuda_module);
    RegisterCudaEngineOptions(&cuda_module);
    RegisterCudaEngine(&cuda_module);
#endif

#ifdef PPLNN_USE_X86
    pybind11::module x86_module = m.def_submodule("x86");
    RegisterX86BuiltinOpImpls();
    RegisterX86EngineFactory(&x86_module);
    RegisterX86EngineOptions(&x86_module);
    RegisterX86Engine(&x86_module);
#endif

#ifdef PPLNN_USE_RISCV
    pybind11::module riscv_module = m.def_submodule("riscv");
    RegisterRiscvBuiltinOpImpls();
    RegisterRiscvEngineFactory(&riscv_module);
    RegisterRiscvEngineOptions(&riscv_module);
    RegisterRiscvEngine(&riscv_module);
#endif

#ifdef PPLNN_USE_ARM
    pybind11::module arm_module = m.def_submodule("arm");
    RegisterArmBuiltinOpImpls();
    RegisterArmEngineFactory(&arm_module);
    RegisterArmEngineOptions(&arm_module);
    RegisterArmEngine(&arm_module);
#endif
}

}}} // namespace ppl::nn::python
