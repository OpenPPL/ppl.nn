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
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace python {

RetCode PyTensor::ConvertFromHost(const pybind11::buffer& b) {
    pybind11::buffer_info info = b.request();

    vector<int64_t> dims(info.ndim);
    for (pybind11::ssize_t i = 0; i < info.ndim; ++i) {
        dims[i] = info.shape[i];
    }

    TensorShape& shape = tensor_->GetShape();
    shape.Reshape(dims);

    TensorShape src_shape = shape;
    src_shape.SetDataFormat(DATAFORMAT_NDARRAY);

    auto status = tensor_->ReallocBuffer();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "realloc buffer of [" << shape.GetBytesIncludingPadding()
                   << "] bytes failed when setting data for tensor[" << tensor_->GetName()
                   << "]: " << GetRetCodeStr(status);
        return status;
    }

    status = tensor_->ConvertFromHost(info.ptr, src_shape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy data to tensor[" << tensor_->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

PyNdArray PyTensor::ConvertToHost() const {
    PyNdArray arr;
    if (tensor_->GetShape().GetBytesExcludingPadding() == 0) {
        return arr;
    }

    TensorShape dst_shape = tensor_->GetShape();
    dst_shape.SetDataFormat(DATAFORMAT_NDARRAY);

    arr.data.resize(dst_shape.GetBytesExcludingPadding());
    auto status = tensor_->ConvertToHost(arr.data.data(), dst_shape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy data of tensor[" << tensor_->GetName() << "] to host failed: " << GetRetCodeStr(status);
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
        .def("GetName", &PyTensor::GetName, pybind11::return_value_policy::reference)
        .def("GetShape", &PyTensor::GetConstShape, pybind11::return_value_policy::reference)
        .def("ConvertFromHost", &PyTensor::ConvertFromHost)
        .def("ConvertToHost", &PyTensor::ConvertToHost, pybind11::return_value_policy::move);
}

}}} // namespace ppl::nn::python
