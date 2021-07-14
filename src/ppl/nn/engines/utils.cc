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

#include "ppl/nn/engines/utils.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

void IrShape2TensorShape(const ir::Shape& ir_shape, TensorShape* tensor_shape) {
    tensor_shape->SetDataType(ir_shape.data_type);
    tensor_shape->SetDataFormat(ir_shape.data_format);
    tensor_shape->Reshape(ir_shape.dims.data(), ir_shape.dims.size());
}

/* -------------------------------------------------------------------------- */

RetCode CopyBuffer(const BufferDesc& src_buf, const TensorShape& src_shape, const Device* src_dev, TensorImpl* dst,
                   Device* tmp_cpu_device) {
    RetCode status;
    auto dst_dev = dst->GetDevice();

    if (dst_dev == src_dev) {
        auto converter = dst_dev->GetDataConverter();
        status = dst->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << dst->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        status = converter->Convert(&dst->GetBufferDesc(), dst->GetShape(), src_buf, src_shape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data to tensor[" << dst->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    } else {
        utils::GenericCpuDevice cpu_device;
        if (!tmp_cpu_device) {
            tmp_cpu_device = &cpu_device;
        }

        BufferDesc tmp_buffer_desc;
        auto status = tmp_cpu_device->Realloc(dst->GetShape(), &tmp_buffer_desc);
        if (status != RC_SUCCESS) {
            return status;
        }
        BufferDescGuard __tmp_buffer_guard__(&tmp_buffer_desc, [tmp_cpu_device](BufferDesc* buffer) {
            tmp_cpu_device->Free(buffer);
        });
        auto tmp_buffer = tmp_buffer_desc.addr;

        auto src_converter = src_dev->GetDataConverter();
        status = src_converter->ConvertToHost(tmp_buffer, dst->GetShape(), src_buf, src_shape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy data to host failed: " << GetRetCodeStr(status);
            return status;
        }

        status = dst->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << dst->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        status = dst->CopyFromHost(tmp_buffer);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy data from host failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

/* -------------------------------------------------------------------------- */

RetCode GenericLoadConstant(edgeid_t eid, const ir::Constant& constant, const TensorShape& shape, Device* device,
                            RuntimeConstantInfo* info) {
    info->SetDevice(device);
    info->Reshape(shape);

    auto status = info->ReallocBuffer();
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
        return status;
    }

    status = device->CopyFromHost(&info->GetBufferDesc(), constant.data.data(), shape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode LoadConstants(const ir::Graph& graph, Device* device, map<edgeid_t, RuntimeConstantInfo>* constants) {
    auto topo = graph.topo.get();
    auto graph_data = graph.data.get();

    for (auto x = graph_data->constants.begin(); x != graph_data->constants.end(); ++x) {
        auto eid = x->first;
        auto edge = topo->GetEdgeById(eid);

        auto shape_ref = graph_data->shapes.find(eid);
        if (shape_ref == graph_data->shapes.end()) {
            LOG(ERROR) << "cannot find shape of constant[" << edge->GetName() << "]";
            return RC_NOT_FOUND;
        }

        TensorShape tensor_shape;
        utils::IrShape2TensorShape(shape_ref->second, &tensor_shape);

        RuntimeConstantInfo constant_info;
        auto status = GenericLoadConstant(eid, x->second, tensor_shape, device, &constant_info);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "load constant[" << edge->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        constants->emplace(eid, std::move(constant_info));
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::utils
