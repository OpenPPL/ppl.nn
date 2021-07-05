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

#include "ppl/nn/engines/common/onnx/split_to_sequence_kernel.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/tensor_sequence.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

void SplitToSequenceKernel::SetExecutionInfo(uint64_t axis, uint64_t keepdims, const SplitFunc& f) {
    axis_ = axis;
    keepdims_ = keepdims;
    split_func_ = f;
}

static bool GetRealAxis(int64_t orig_axis, uint32_t dim_count, int64_t* axis) {
    if (orig_axis < 0) {
        *axis = dim_count + orig_axis;
    } else {
        *axis = orig_axis;
    }

    return (*axis >= 0 && *axis < dim_count);
}

static bool CheckSplitChunks(const vector<int64_t>& chunks, uint32_t orig_axis_dim) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < chunks.size(); ++i) {
        sum += chunks[i];
    }
    return (sum == orig_axis_dim);
}

static RetCode GetSplitChunks(const TensorImpl* split, uint32_t orig_axis_dim, vector<int64_t>* split_chunks) {
    if (!split) {
        split_chunks->resize(orig_axis_dim, 1);
    } else {
        auto split_data_type = split->GetShape().GetDataType();

        if (split->GetShape().IsScalar()) {
            int64_t dims_of_chunk;
            if (split_data_type == DATATYPE_INT32) {
                int32_t tmp = 0;
                auto status = split->CopyToHost(&tmp);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "get split data failed: " << GetRetCodeStr(status);
                    return status;
                }
                dims_of_chunk = tmp;
            } else if (split_data_type == DATATYPE_INT64) {
                auto status = split->CopyToHost(&dims_of_chunk);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "get split data failed: " << GetRetCodeStr(status);
                    return status;
                }
            } else {
                LOG(ERROR) << "invalid split data type[" << GetDataTypeStr(split_data_type) << "]";
                return RC_INVALID_VALUE;
            }

            auto nr_chunk = orig_axis_dim / dims_of_chunk;
            split_chunks->resize(nr_chunk, dims_of_chunk);

            auto remaining = orig_axis_dim % dims_of_chunk;
            if (remaining != 0) {
                split_chunks->push_back(remaining);
            }
        } else {
            auto nr_chunk = split->GetShape().GetElementsExcludingPadding();
            if (split_data_type == DATATYPE_INT32) {
                vector<int32_t> chunks(nr_chunk);
                auto status = split->CopyToHost(chunks.data());
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "get split chunks failed: " << GetRetCodeStr(status);
                    return status;
                }

                split_chunks->resize(nr_chunk);
                for (uint32_t i = 0; i < nr_chunk; ++i) {
                    split_chunks->at(i) = chunks[i];
                }
            } else if (split_data_type == DATATYPE_INT64) {
                split_chunks->resize(nr_chunk);
                auto status = split->CopyToHost(split_chunks->data());
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "get split chunks failed: " << GetRetCodeStr(status);
                    return status;
                }
            } else {
                LOG(ERROR) << "invalid split data type[" << GetDataTypeStr(split_data_type) << "]";
                return RC_INVALID_VALUE;
            }

            if (!CheckSplitChunks(*split_chunks, orig_axis_dim)) {
                LOG(ERROR) << "sum of split chunks from `split` is not equal to dims[" << orig_axis_dim << "]";
                return RC_INVALID_VALUE;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode InitNdarrayBufferInfo(const TensorShape& inshape, const BufferDesc& inbuf, Device* device,
                                     TensorBufferInfo* info) {
    info->SetDevice(device);
    info->Reshape(inshape);
    info->GetShape().SetDataFormat(DATAFORMAT_NDARRAY);

    auto status = info->ReallocBuffer();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc [" << info->GetShape().GetBytesIncludingPadding()
                   << "] bytes for tmp buffer failed: " << GetRetCodeStr(status);
        return status;
    }

    status = device->GetDataConverter()->Convert(&info->GetBufferDesc(), info->GetShape(), inbuf, inshape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "convert data format from [" << GetDataFormatStr(inshape.GetDataFormat())
                   << "] to [NDARRAY] failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

static RetCode CheckInputDataType(TensorImpl* input0) {
    auto data_type = input0->GetShape().GetDataType();
    if (data_type == DATATYPE_FLOAT16 || data_type == DATATYPE_FLOAT32 || data_type == DATATYPE_FLOAT64 ||
        data_type == DATATYPE_INT16 || data_type == DATATYPE_INT32 || data_type == DATATYPE_INT64) {
        return RC_SUCCESS;
    }

    LOG(ERROR) << "unsupported input data type[" << GetDataTypeStr(data_type) << "]";
    return RC_UNSUPPORTED;
}

RetCode SplitToSequenceKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto split = ctx->GetInput<TensorImpl>(1);
    auto device = GetDevice();

    auto status = CheckInputDataType(input);
    if (status != RC_SUCCESS) {
        return status;
    }

    const BufferDesc* src_buffer = &input->GetBufferDesc();
    const TensorShape* src_shape = &input->GetShape();
    TensorBufferInfo ndarray_buffer_info;
    if (input->GetShape().GetDataFormat() != DATAFORMAT_NDARRAY) {
        status = InitNdarrayBufferInfo(input->GetShape(), input->GetBufferDesc(), device, &ndarray_buffer_info);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "init tmp ndarray buffer failed: " << GetRetCodeStr(status);
            return status;
        }

        src_buffer = &ndarray_buffer_info.GetBufferDesc();
        src_shape = &ndarray_buffer_info.GetShape();
    }

    int64_t axis = 0;
    if (!GetRealAxis(axis_, src_shape->GetDimCount(), &axis)) {
        LOG(ERROR) << "invalid orig axis[" << axis_ << "], real axis[" << axis << "], dimcount["
                   << src_shape->GetDimCount() << "]";
        return RC_INVALID_VALUE;
    }

    const uint32_t orig_axis_dim = src_shape->GetDim(axis);

    vector<int64_t> split_chunks;
    status = GetSplitChunks(split, orig_axis_dim, &split_chunks);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GetSplitChunks of axis[" << axis << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    const uint64_t dims_before_axis = src_shape->GetElementsToDimensionExcludingPadding(axis);
    const uint64_t dims_from_axis = src_shape->GetElementsFromDimensionExcludingPadding(axis);
    const uint64_t dims_after_axis = src_shape->GetElementsFromDimensionExcludingPadding(axis + 1);
    const uint32_t element_size = GetSizeOfDataType(src_shape->GetDataType());

    auto input_buffer_cursor = *src_buffer;
    auto output = ctx->GetOutput<TensorSequence>(0);
    for (uint32_t chunk_idx = 0; chunk_idx < split_chunks.size(); ++chunk_idx) {
        auto dims_of_chunk = split_chunks[chunk_idx];

        TensorBufferInfo buffer_info;
        buffer_info.SetDevice(device);
        buffer_info.Reshape(*src_shape);
        buffer_info.GetShape().SetDim(axis, dims_of_chunk);
        status = buffer_info.ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "alloc tensor buffer failed, bytes[" << buffer_info.GetShape().GetBytesIncludingPadding()
                       << "]";
            return status;
        }

        status = split_func_(dims_before_axis, dims_from_axis, dims_after_axis, dims_of_chunk, element_size, device,
                             &input_buffer_cursor, &buffer_info.GetBufferDesc());
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "split data failed: " << GetRetCodeStr(status);
            return status;
        }

        if (!split && keepdims_ == 0) {
            const TensorShape& orig_shape = buffer_info.GetShape();
            vector<int64_t> new_dims;
            new_dims.reserve(orig_shape.GetDimCount());
            for (uint32_t i = 0; i < orig_shape.GetDimCount(); ++i) {
                if (i != axis) {
                    new_dims.push_back(orig_shape.GetDim(i));
                }
            }

            buffer_info.GetShape().Reshape(new_dims);
        }

        output->EmplaceBack(std::move(buffer_info));
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::common
