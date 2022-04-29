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
#include "ppl/nn/utils/destructor.h"
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

        status = converter->Convert(&dst->GetBufferDesc(), *dst->GetShape(), src_buf, src_shape);
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
        auto status = tmp_cpu_device->Realloc(*dst->GetShape(), &tmp_buffer_desc);
        if (status != RC_SUCCESS) {
            return status;
        }
        utils::Destructor __tmp_buffer_guard__([tmp_cpu_device, &tmp_buffer_desc]() -> void {
            tmp_cpu_device->Free(&tmp_buffer_desc);
        });
        auto tmp_buffer = tmp_buffer_desc.addr;

        auto src_converter = src_dev->GetDataConverter();
        status = src_converter->ConvertToHost(tmp_buffer, *dst->GetShape(), src_buf, src_shape);
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

RetCode GenericLoadConstant(const void* data, uint64_t size, const TensorShape& shape, Device* device,
                            RuntimeConstantInfo* info, bool omit_data) {
    info->Reshape(shape);

    if (!omit_data) {
        info->SetDevice(device);
        auto status = info->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }

        status = device->CopyFromHost(&info->GetBufferDesc(), data, shape);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

RetCode GenericLoadConstant(const void* data, uint64_t size, const TensorShape& shape, Device* device,
                            BufferInfo* info) {
    info->SetDevice(device);
    auto status = info->ReallocBuffer(shape);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
        return status;
    }

    status = device->CopyFromHost(&info->GetBufferDesc(), data, size);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy constant failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode LoadConstants(const ir::Graph& graph, Device* device, map<edgeid_t, RuntimeConstantInfo>* constants,
                      const std::set<edgeid_t>* data_omitted_constants) {
    auto topo = graph.topo.get();
    auto graph_data = graph.data.get();

    for (uint32_t i = 0; i < topo->GetConstantCount(); ++i) {
        auto eid = topo->GetConstant(i);
        auto edge = topo->GetEdge(eid);
        if (edge == nullptr) {
            LOG(ERROR) << "cannot get edge of constant[edgeid=" << eid << "]";
            return RC_NOT_FOUND;
        }

        auto ret_pair = constants->insert(make_pair(eid, RuntimeConstantInfo()));
        if (!ret_pair.second) {
            continue;
        }

        auto shape_ref = graph_data->shapes.find(eid);
        if (shape_ref == graph_data->shapes.end()) {
            LOG(ERROR) << "cannot find shape of constant[" << edge->GetName() << "]";
            return RC_NOT_FOUND;
        }

        TensorShape tensor_shape;
        utils::IrShape2TensorShape(shape_ref->second, &tensor_shape);

        auto constant_ref = graph_data->constants.find(eid);
        if (constant_ref == graph_data->constants.end()) {
            LOG(ERROR) << "cannot find data of constant[" << edge->GetName() << "]";
            return RC_NOT_FOUND;
        }

        bool omit_data = false;
        if (data_omitted_constants != nullptr) {
            omit_data = (data_omitted_constants->find(eid) != data_omitted_constants->end());
        }

        RuntimeConstantInfo& constant_info = ret_pair.first->second;
        auto status = GenericLoadConstant(constant_ref->second.data.data(), constant_ref->second.data.size(),
                                          tensor_shape, device, &constant_info, omit_data);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "load constant[" << edge->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

RetCode LoadConstants(const ConstantVisitor& visitor, Device* dev, map<edgeid_t, BufferInfo>* eid2info) {
    return visitor.ForEach(
        [eid2info, dev](const ir::Edge* edge, const void* data, uint64_t size, const TensorShape& shape) -> RetCode {
            BufferInfo info;
            auto status = utils::GenericLoadConstant(data, size, shape, dev, &info);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "load constant failed: " << GetRetCodeStr(status);
                return status;
            }

            auto ret_pair = eid2info->emplace(edge->GetId(), std::move(info));
            if (!ret_pair.second) {
                LOG(ERROR) << "constant[" << edge->GetName() << "] already exists.";
                return RC_EXISTS;
            }
            return RC_SUCCESS;
        });
}

}}} // namespace ppl::nn::utils
