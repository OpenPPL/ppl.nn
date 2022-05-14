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
namespace cuda {
void RegisterBuiltinOpImpls();
void RegisterEngineFactory(pybind11::module*);
void RegisterEngineOptions(pybind11::module*);
void RegisterEngine(pybind11::module*);
} // namespace cuda
#endif

#ifdef PPLNN_USE_X86
namespace x86 {
void RegisterBuiltinOpImpls();
void RegisterEngineFactory(pybind11::module*);
void RegisterEngineOptions(pybind11::module*);
void RegisterEngine(pybind11::module*);
} // namespace x86
#endif

#ifdef PPLNN_USE_RISCV
namespace riscv {
void RegisterBuiltinOpImpls();
void RegisterEngineFactory(pybind11::module*);
void RegisterEngineOptions(pybind11::module*);
void RegisterEngine(pybind11::module*);
} // namespace riscv
#endif

#ifdef PPLNN_USE_ARM
namespace arm {
void RegisterBuiltinOpImpls();
void RegisterEngineFactory(pybind11::module*);
void RegisterEngineOptions(pybind11::module*);
void RegisterEngine(pybind11::module*);
} // namespace arm
#endif

void RegisterTensorShape(pybind11::module*);
void RegisterTensor(pybind11::module*);
void RegisterNdArray(pybind11::module*);
void RegisterEngine(pybind11::module*);
void RegisterDeviceContext(pybind11::module*);
void RegisterRuntime(pybind11::module*);
void RegisterVersion(pybind11::module*);

#ifdef PPLNN_ENABLE_ONNX_MODEL
namespace onnx {
void RegisterRuntimeBuilderResources(pybind11::module*);
void RegisterRuntimeBuilder(pybind11::module*);
void RegisterRuntimeBuilderFactory(pybind11::module*);
} // namespace onnx
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
namespace pmx {
void RegisterRuntimeBuilderResources(pybind11::module*);
void RegisterRuntimeBuilder(pybind11::module*);
void RegisterRuntimeBuilderFactory(pybind11::module*);
} // namespace pmx
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
    onnx::RegisterRuntimeBuilderResources(&onnx_module);
    onnx::RegisterRuntimeBuilderFactory(&onnx_module);
    onnx::RegisterRuntimeBuilder(&onnx_module);
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
    pybind11::module pmx_module = m.def_submodule("pmx");
    pmx::RegisterRuntimeBuilderResources(&pmx_module);
    pmx::RegisterRuntimeBuilderFactory(&pmx_module);
    pmx::RegisterRuntimeBuilder(&pmx_module);
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
    cuda::RegisterBuiltinOpImpls();
    cuda::RegisterEngineFactory(&cuda_module);
    cuda::RegisterEngineOptions(&cuda_module);
    cuda::RegisterEngine(&cuda_module);
#endif

#ifdef PPLNN_USE_X86
    pybind11::module x86_module = m.def_submodule("x86");
    x86::RegisterBuiltinOpImpls();
    x86::RegisterEngineFactory(&x86_module);
    x86::RegisterEngineOptions(&x86_module);
    x86::RegisterEngine(&x86_module);
#endif

#ifdef PPLNN_USE_RISCV
    pybind11::module riscv_module = m.def_submodule("riscv");
    riscv::RegisterBuiltinOpImpls();
    riscv::RegisterEngineFactory(&riscv_module);
    riscv::RegisterEngineOptions(&riscv_module);
    riscv::RegisterEngine(&riscv_module);
#endif

#ifdef PPLNN_USE_ARM
    pybind11::module arm_module = m.def_submodule("arm");
    arm::RegisterBuiltinOpImpls();
    arm::RegisterEngineFactory(&arm_module);
    arm::RegisterEngineOptions(&arm_module);
    arm::RegisterEngine(&arm_module);
#endif
}

}}} // namespace ppl::nn::python
