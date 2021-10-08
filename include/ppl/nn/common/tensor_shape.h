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

#ifndef _ST_HPC_PPL_NN_COMMON_TENSOR_SHAPE_H_
#define _ST_HPC_PPL_NN_COMMON_TENSOR_SHAPE_H_

#include "ppl/common/types.h"
#include "ppl/nn/common/common.h"
#include <vector>

namespace ppl { namespace nn {

class PPLNN_PUBLIC TensorShape final {
private:
    static const uint32_t kAxisC = 1;

private:
    bool is_scalar_;
    ppl::common::datatype_t data_type_;
    ppl::common::dataformat_t data_format_;
    std::vector<int64_t> dims_;
    std::vector<uint16_t> padding0_;
    std::vector<uint16_t> padding1_;

private:
    static int16_t CalcPadding(int64_t dim, uint32_t alignment) {
        return static_cast<int16_t>((((uintptr_t)dim + (uintptr_t)alignment - 1) & ~((uintptr_t)alignment - 1)) -
                                    (uintptr_t)dim);
    }

public:
    TensorShape()
        : is_scalar_(false), data_type_(ppl::common::DATATYPE_UNKNOWN), data_format_(ppl::common::DATAFORMAT_UNKNOWN) {}

    TensorShape(const TensorShape& other) = default;
    TensorShape& operator=(const TensorShape& other) = default;

    uint32_t GetRealDimCount() const {
        return dims_.size();
    };
    uint32_t GetDimCount() const {
        if (is_scalar_) {
            return 1;
        }
        return dims_.size();
    }
    int64_t GetDim(uint32_t which) const {
        if (is_scalar_) {
            return 1;
        }
        return dims_[which];
    }
    const int64_t* GetDims() const {
        return dims_.data();
    }
    uint16_t GetPadding0(uint32_t which) const {
        if (is_scalar_) {
            return 0;
        }
        return padding0_[which];
    }
    uint16_t GetPadding1(uint32_t which) const {
        if (is_scalar_) {
            return 0;
        }
        return padding1_[which];
    }
    const uint16_t* GetPadding0s() const {
        return padding0_.data();
    }
    const uint16_t* GetPadding1s() const {
        return padding1_.data();
    }

    ppl::common::datatype_t GetDataType() const {
        return data_type_;
    }
    ppl::common::dataformat_t GetDataFormat() const {
        return data_format_;
    }
    void SetDataType(ppl::common::datatype_t dt) {
        data_type_ = dt;
    }
    void CalcPadding() {
        if (data_format_ == ppl::common::DATAFORMAT_NDARRAY) {
            for (uint32_t i = 0; i < dims_.size(); ++i) {
                padding0_[i] = 0;
                padding1_[i] = 0;
            }
        } else if (dims_.size() >= 2) {
            if (data_format_ == ppl::common::DATAFORMAT_N2CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 2);
            } else if (data_format_ == ppl::common::DATAFORMAT_N4CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 4);
            } else if (data_format_ == ppl::common::DATAFORMAT_N8CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 8);
            } else if (data_format_ == ppl::common::DATAFORMAT_N16CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 16);
            } else if (data_format_ == ppl::common::DATAFORMAT_N32CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 32);
            } else if (data_format_ == ppl::common::DATAFORMAT_N16CX) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 16);
            } else if (data_format_ == ppl::common::DATAFORMAT_NHWC8) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 8);
            } else if (data_format_ == ppl::common::DATAFORMAT_NHWC16) {
                padding1_[TensorShape::kAxisC] = CalcPadding(dims_[1], 16);
            }
        }
    }

    void SetDataFormat(ppl::common::dataformat_t data_format) {
        // do not change padding if data format is not changed
        if (data_format != data_format_) {
            data_format_ = data_format;
            CalcPadding();
        }
    }

    void ReshapeAsScalar() {
        dims_.clear();
        padding0_.clear();
        padding1_.clear();
        is_scalar_ = true;
    }

    void Reshape(const int64_t* dims, uint32_t dim_count) {
        DoResize(dim_count);
        if (dim_count == 0) {
            is_scalar_ = true;
            return;
        }

        is_scalar_ = false;
        for (uint32_t i = 0; i < dim_count; ++i) {
            dims_[i] = dims[i];
        }
        CalcPadding();
    }

    void Reshape(const std::vector<int64_t>& dims) {
        return Reshape(dims.data(), dims.size());
    }

    void SetPadding0(uint32_t which, uint16_t padding) {
        padding0_[which] = padding;
    }
    void SetPadding1(uint32_t which, uint16_t padding) {
        padding1_[which] = padding;
    }
    void SetDim(uint32_t which, int64_t dim) {
        dims_[which] = dim;
    }
    void SetDimCount(uint32_t dc) {
        DoResize(dc);
        is_scalar_ = (dc == 0);
    }

    uint64_t GetElementsIncludingPadding() const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < dims_.size(); ++i) {
            accu *= (dims_[i] + padding0_[i] + padding1_[i]);
        }
        return accu;
    }
    uint64_t GetElementsExcludingPadding() const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < dims_.size(); ++i) {
            accu *= dims_[i];
        }
        return accu;
    }
    uint64_t GetBytesIncludingPadding() const {
        return ppl::common::GetSizeOfDataType(data_type_) * GetElementsIncludingPadding();
    }
    uint64_t GetBytesExcludingPadding() const {
        return ppl::common::GetSizeOfDataType(data_type_) * GetElementsExcludingPadding();
    }
    uint64_t GetElementsFromDimensionIncludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = which; i < dims_.size(); ++i) {
            accu *= (dims_[i] + padding0_[i] + padding1_[i]);
        }
        return accu;
    }
    uint64_t GetElementsToDimensionIncludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < which; ++i) {
            accu *= (dims_[i] + padding0_[i] + padding1_[i]);
        }
        return accu;
    }
    uint64_t GetElementsFromDimensionExcludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = which; i < dims_.size(); ++i) {
            accu *= dims_[i];
        }
        return accu;
    }
    uint64_t GetElementsToDimensionExcludingPadding(uint32_t which) const {
        if (dims_.empty()) {
            return is_scalar_ ? 1 : 0;
        }
        uint64_t accu = 1;
        for (uint32_t i = 0; i < which; ++i) {
            accu *= dims_[i];
        }
        return accu;
    }
    uint64_t GetBytesToDimesionIncludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * GetElementsToDimensionIncludingPadding(which);
    }
    uint64_t GetBytesToDimesionExcludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * GetElementsToDimensionExcludingPadding(which);
    }
    uint64_t GetBytesFromDimesionIncludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * GetElementsFromDimensionIncludingPadding(which);
    }
    uint64_t GetBytesFromDimesionExcludingPadding(uint32_t which) const {
        return ppl::common::GetSizeOfDataType(data_type_) * GetElementsFromDimensionExcludingPadding(which);
    }

    void Clear() {
        is_scalar_ = false;
        data_type_ = ppl::common::DATATYPE_UNKNOWN;
        data_format_ = ppl::common::DATAFORMAT_UNKNOWN;
        dims_.clear();
        padding0_.clear();
        padding1_.clear();
    }

    bool IsScalar() const {
        return is_scalar_;
    }

    bool IsEmpty() const {
        if (is_scalar_) {
            return false;
        }
        if (dims_.empty()) {
            return true;
        }

        uint64_t accu = 1;
        for (uint32_t i = 0; i < dims_.size(); ++i) {
            accu *= dims_[i];
        }
        return (accu == 0);
    }

private:
    void DoResize(uint32_t dim_count) {
        dims_.resize(dim_count, 0);
        padding0_.resize(dim_count, 0);
        padding1_.resize(dim_count, 0);
    }
};

}} // namespace ppl::nn

#endif
