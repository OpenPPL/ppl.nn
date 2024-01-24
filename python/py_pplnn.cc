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

#ifdef PPLNN_USE_CUDA
#include "engines/cuda/py_cuda.h"
#endif

#ifdef PPLNN_USE_X86
#include "engines/x86/py_x86.h"
#endif

#ifdef PPLNN_USE_RISCV
#include "engines/riscv/py_riscv.h"
#endif

#ifdef PPLNN_USE_ARM
#include "engines/arm/py_arm.h"
#endif

#ifdef PPLNN_ENABLE_ONNX_MODEL
#include "models/onnx/py_onnx.h"
#endif

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "models/pmx/py_pmx.h"
#endif

namespace ppl { namespace nn { namespace python {

void RegisterLogger(pybind11::module*);
void RegisterTensorShape(pybind11::module*);
void RegisterTensor(pybind11::module*);
void RegisterNdArray(pybind11::module*);
void RegisterEngine(pybind11::module*);
void RegisterDeviceContext(pybind11::module*);
void RegisterRuntime(pybind11::module*);
void RegisterVersion(pybind11::module*);
void RegisterModelOptionsBase(pybind11::module*);
void LoadResources(pybind11::module*);

PYBIND11_MODULE(nn, m) {
    RegisterLogger(&m);
    RegisterTensorShape(&m);
    RegisterTensor(&m);
    RegisterNdArray(&m);
    RegisterEngine(&m);
    RegisterDeviceContext(&m);
    RegisterRuntime(&m);
    RegisterVersion(&m);
    RegisterModelOptionsBase(&m);

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
    pmx::RegisterLoadModelOptions(&pmx_module);
    pmx::RegisterSaveModelOptions(&pmx_module);
#endif

#ifdef PPLNN_USE_CUDA
    pybind11::module cuda_module = m.def_submodule("cuda");
    cuda::RegisterEngineFactory(&cuda_module);
    cuda::RegisterEngineOptions(&cuda_module);
    cuda::RegisterEngine(&cuda_module);
#endif

#ifdef PPLNN_USE_X86
    pybind11::module x86_module = m.def_submodule("x86");
    x86::RegisterEngineFactory(&x86_module);
    x86::RegisterEngineOptions(&x86_module);
    x86::RegisterEngine(&x86_module);
#endif

#ifdef PPLNN_USE_RISCV
    pybind11::module riscv_module = m.def_submodule("riscv");
    riscv::RegisterEngineFactory(&riscv_module);
    riscv::RegisterEngineOptions(&riscv_module);
    riscv::RegisterEngine(&riscv_module);
#endif

#ifdef PPLNN_USE_ARM
    pybind11::module arm_module = m.def_submodule("arm");
    arm::RegisterEngineFactory(&arm_module);
    arm::RegisterEngineOptions(&arm_module);
    arm::RegisterEngine(&arm_module);
#endif

    LoadResources(&m);
}

}}} // namespace ppl::nn::python
