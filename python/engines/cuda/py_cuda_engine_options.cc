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

#ifdef PPLNN_USE_CUDA

#include "ppl/nn/engines/cuda/cuda_engine_options.h"
#include "pybind11/pybind11.h"

namespace ppl { namespace nn { namespace python {

void RegisterCudaEngineOptions(pybind11::module* m) {
    pybind11::class_<CudaEngineOptions>(*m, "CudaEngineOptions")
        .def(pybind11::init<>())
        .def_readwrite("device_id", &CudaEngineOptions::device_id)
        .def_readwrite("mm_policy", &CudaEngineOptions::mm_policy);

    m->attr("CUDA_MM_COMPACT") = (uint32_t)CUDA_MM_COMPACT;
    m->attr("CUDA_MM_BEST_FIT") = (uint32_t)CUDA_MM_BEST_FIT;
}

}}} // namespace ppl::nn::python

#endif
