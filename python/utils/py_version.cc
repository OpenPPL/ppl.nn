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

namespace ppl { namespace nn { namespace python {

void RegisterVersion(pybind11::module* m) {
    m->attr("PPLNN_VERSION_MAJOR") = (uint32_t)PPLNN_VERSION_MAJOR;
    m->attr("PPLNN_VERSION_MINOR") = (uint32_t)PPLNN_VERSION_MINOR;
    m->attr("PPLNN_VERSION_PATCH") = (uint32_t)PPLNN_VERSION_PATCH;
    m->attr("PPLNN_COMMIT_STR") = PPLNN_COMMIT_STR;
}

}}} // namespace ppl::nn::python
