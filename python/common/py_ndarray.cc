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

#include "py_ndarray.h"
#include "pybind11/pybind11.h"

namespace ppl { namespace nn { namespace python {

static const char* g_datatype2format[] = {
    "c", // DATATYPE_UNKNOWN -> char
    "B", // DATATYPE_UINT8 -> unsigned char
    "H", // DATATYPE_UINT16 -> unsigned short
    "I", // DATATYPE_UINT32 -> unsigned int
    "L", // DATATYPE_UINT64 -> unsigned long
    "e", // DATATYPE_FLOAT16 -> 2 bytes
    "f", // DATATYPE_FLOAT32 -> float
    "d", // DATATYPE_FLOAT64 -> double
    "e", // DATATYPE_BFLOAT16 -> 2 bytes
    "b", // DATATYPE_INT4B -> signed char
    "b", // DATATYPE_INT8 -> signed char
    "h", // DATATYPE_INT16 -> short
    "i", // DATATYPE_INT32 -> int
    "l", // DATATYPE_INT64 -> long
    "?", // DATATYPE_BOOL -> unsigned char
    "L", // DATATYPE_COMPLEX64 -> 8 bytes
    "s", // DATATYPE_COMPLEX128 -> 16 bytes
};

void RegisterNdArray(pybind11::module* m) {
    pybind11::class_<PyNdArray>(*m, "NdArray", pybind11::buffer_protocol())
        .def("__bool__",
             [](const PyNdArray& arr) -> bool {
                 return (!arr.data.empty());
             })
        .def_buffer([](PyNdArray& arr) -> pybind11::buffer_info {
            return pybind11::buffer_info(arr.data.data(), ppl::common::GetSizeOfDataType(arr.data_type),
                                         g_datatype2format[arr.data_type], arr.dims.size(), arr.dims, arr.strides);
        });
}

}}} // namespace ppl::nn::python
