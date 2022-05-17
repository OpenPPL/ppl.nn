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

#include "py_tensor.h"
#include "../common/py_device_context.h"
#include "ppl/nn/common/logger.h"
#include <map>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

static const map<string, datatype_t> g_format2datatype = {
    {"B", DATATYPE_UINT8}, // -> unsigned char
    {"H", DATATYPE_UINT16}, //  -> unsigned short
    {"I", DATATYPE_UINT32}, //  -> unsigned int
    {"L", DATATYPE_UINT64}, //  -> unsigned long
    {"e", DATATYPE_FLOAT16}, //  -> 2 bytes
    {"f", DATATYPE_FLOAT32}, //  -> float
    {"d", DATATYPE_FLOAT64}, //  -> double
    {"b", DATATYPE_INT8}, //  -> signed char
    {"h", DATATYPE_INT16}, //  -> short
    {"i", DATATYPE_INT32}, //  -> int
    {"l", DATATYPE_INT64}, //  -> long
    {"?", DATATYPE_BOOL}, //  -> unsigned char
};

static RetCode ConvertFromHost(const PyTensor& py_tensor, const pybind11::buffer& b) {
    auto tensor = py_tensor.ptr;
    pybind11::buffer_info info = b.request();

    vector<int64_t> dims(info.ndim);
    for (pybind11::ssize_t i = 0; i < info.ndim; ++i) {
        dims[i] = info.shape[i];
    }

    auto shape = tensor->GetShape();
    shape->Reshape(dims);

    auto ref = g_format2datatype.find(info.format);
    if (ref == g_format2datatype.end()) {
        LOG(ERROR) << "unsupported data format[\"" << info.format << "\"]";
        return RC_UNSUPPORTED;
    }
    auto data_type = ref->second;
    LOG(DEBUG) << "data type of input for tensor[" << tensor->GetName() << "] is [" << GetDataTypeStr(data_type)
               << "].";

    TensorShape src_shape = *shape;
    src_shape.SetDataFormat(DATAFORMAT_NDARRAY);
    src_shape.SetDataType(data_type);

    auto status = tensor->ReallocBuffer();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "realloc buffer of [" << shape->GetBytesIncludingPadding()
                   << "] bytes failed when setting data for tensor[" << tensor->GetName()
                   << "]: " << GetRetCodeStr(status);
        return status;
    }

    status = tensor->ConvertFromHost(info.ptr, src_shape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy data to tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

// use original data type and format if `datatype` or `dataformat` are unknown
static PyNdArray ConvertToHost(const PyTensor& py_tensor, datatype_t data_type, dataformat_t data_format) {
    auto tensor = py_tensor.ptr;
    PyNdArray arr;
    if (tensor->GetShape()->GetBytesExcludingPadding() == 0) {
        return arr;
    }

    TensorShape dst_shape = *tensor->GetShape();
    if (data_type != DATATYPE_UNKNOWN) {
        dst_shape.SetDataType(data_type);
    }
    if (data_format != DATAFORMAT_UNKNOWN) {
        dst_shape.SetDataFormat(data_format);
    }

    arr.data.resize(dst_shape.GetBytesExcludingPadding());
    auto status = tensor->ConvertToHost(arr.data.data(), dst_shape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy data of tensor[" << tensor->GetName() << "] to host failed: " << GetRetCodeStr(status);
        return arr;
    }

    arr.data_type = dst_shape.GetDataType();

    auto dim_count = dst_shape.GetRealDimCount();

    arr.dims.resize(dim_count);
    for (uint32_t i = 0; i < dim_count; ++i) {
        arr.dims[i] = dst_shape.GetDim(i);
    }

    arr.strides.resize(dim_count);
    for (uint32_t i = 1; i < dim_count; ++i) {
        arr.strides[i - 1] = dst_shape.GetBytesFromDimesionExcludingPadding(i);
    }
    arr.strides[dim_count - 1] = GetSizeOfDataType(dst_shape.GetDataType());

    return arr;
}

void RegisterTensor(pybind11::module* m) {
    pybind11::class_<PyTensor>(*m, "Tensor")
        .def("__bool__",
             [](const PyTensor& tensor) -> bool {
                 return (tensor.ptr);
             })
        .def("GetBufferPtr",
             [](const PyTensor& tensor) -> uint64_t {
                 return (uint64_t)(tensor.ptr->GetBufferPtr());
             })
        .def("SetBufferPtr",
             [](PyTensor& tensor, uint64_t ptr) -> void {
                 tensor.ptr->SetBufferPtr((void*)ptr);
             })
        .def("GetDeviceContext",
             [](const PyTensor& tensor) -> PyDeviceContext {
                 return PyDeviceContext(tensor.ptr->GetDeviceContext());
             })
        .def("GetName",
             [](const PyTensor& tensor) -> const char* {
                 return tensor.ptr->GetName();
             },
             pybind11::return_value_policy::reference)
        .def("GetShape",
             [](const PyTensor& tensor) -> const TensorShape& {
                 return *tensor.ptr->GetShape();
             },
             pybind11::return_value_policy::reference)
        .def("ConvertFromHost", &ConvertFromHost)
        .def("ConvertToHost", &ConvertToHost, pybind11::return_value_policy::move,
             pybind11::arg("datatype") = (ppl::common::datatype_t)ppl::common::DATATYPE_UNKNOWN,
             pybind11::arg("dataformat") = (ppl::common::dataformat_t)ppl::common::DATAFORMAT_NDARRAY);
}

}}} // namespace ppl::nn::python
