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

#include "ppl/nn/common/tensor_buffer_info.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/engines/common/onnx/loop_kernel.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

template <typename T>
void DummyDeleter(T*) {}

RetCode LoopKernel::SetExecutionInfo(const shared_ptr<ir::GraphTopo>& topo, const RuntimeGraphInfo* info,
                                     const RuntimeAuxInfo* aux_info, const RuntimeInitInfo* init_info,
                                     LoopConcatOutputFunc func) {
    auto status =
        subgraph_.Init(topo, shared_ptr<const RuntimeGraphInfo>(info, DummyDeleter<const RuntimeGraphInfo>),
                       shared_ptr<const RuntimeAuxInfo>(aux_info, DummyDeleter<const RuntimeAuxInfo>), *init_info);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init loop kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    concat_output_func_ = func;
    return RC_SUCCESS;
}

static inline int64_t GetMaxTripCount(const KernelExecContext& ctx) {
    int64_t trip_count = INT64_MAX;
    auto input0 = ctx.GetInput<TensorImpl>(0);
    if (input0) {
        input0->CopyToHost(&trip_count);
    }
    return trip_count;
}

static inline RetCode GetKeepGoing(const KernelExecContext& ctx, bool* keep_going) {
    *keep_going = true;
    auto input1 = ctx.GetInput<TensorImpl>(1);
    if (input1) {
        auto status = input1->CopyToHost(keep_going);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "get `keep_going` failed: " << GetRetCodeStr(status);
            return status;
        }
    }
    return RC_SUCCESS;
}

struct LoopInfo final {
    LoopInfo(const KernelExecContext& ctx) {
        loop_carried_dep_num = ctx.GetInputCount() - 2; // N
        scan_output_num = ctx.GetOutputCount() - loop_carried_dep_num; // K
        intermediate_outputs.resize(scan_output_num);
    }

    uint32_t loop_carried_dep_num;
    uint32_t scan_output_num;
    vector<vector<TensorBufferInfo>> intermediate_outputs;
};

static RetCode SaveSubgraphOutputs(RuntimeImpl* subgraph, LoopInfo* info) {
    for (uint32_t i = 0; i < info->scan_output_num; ++i) {
        auto scan_output = subgraph->GetOutputTensorImpl(info->loop_carried_dep_num + i + 1); // +1 for skipping `cond`

        TensorBufferInfo output_buffer;
        if (scan_output->IsBufferOwner()) {
            output_buffer.Reshape(*scan_output->GetShape());
            output_buffer.SetBuffer(scan_output->GetBufferDesc(), scan_output->GetDevice(), true);
            scan_output->DetachBuffer();
        } else {
            auto device = scan_output->GetDevice();

            output_buffer.SetDevice(scan_output->GetDevice());
            output_buffer.Reshape(*scan_output->GetShape());
            auto status = output_buffer.ReallocBuffer();
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "alloc buffer for tensor[" << scan_output->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                return status;
            }

            status =
                device->Copy(&output_buffer.GetBufferDesc(), scan_output->GetBufferDesc(), *scan_output->GetShape());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy data from tensor[" << scan_output->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }

        info->intermediate_outputs[i].emplace_back(std::move(output_buffer));
    }

    return RC_SUCCESS;
}

static RetCode UpdateSubgraphInputs(int64_t trip_count, RuntimeImpl* subgraph,
                                    utils::GenericCpuDevice* tmp_cpu_device) {
    auto trip_count_tensor = subgraph->GetInputTensorImpl(0);
    if (trip_count_tensor->GetDevice()) { // not used by anyone if device is not set
        auto status = trip_count_tensor->CopyFromHost(&trip_count);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "get trip count failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    // starting from 1 to skip trip count
    for (uint32_t i = 1; i < subgraph->GetInputCount(); ++i) {
        auto dst = subgraph->GetInputTensorImpl(i);
        if (!dst->GetDevice()) { // not used by anyone if device is not set
            continue;
        }

        auto src = subgraph->GetOutputTensorImpl(i - 1);
        *dst->GetShape() = *src->GetShape();
        if (dst->GetDevice() == src->GetDevice()) {
            dst->TransferBufferFrom(src);
        } else {
            auto status = utils::CopyTensorBuffer(*src, dst, tmp_cpu_device);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy data from tensor[" << src->GetName() << "] to tensor[" << dst->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode InitSubgraphInputs(const KernelExecContext& ctx, bool keep_going,
                                  utils::GenericCpuDevice* tmp_cpu_device, RuntimeImpl* subgraph) {
    RetCode status;

    auto input0 = subgraph->GetInputTensorImpl(0);
    if (input0->GetDevice()) { // not used by anyone if device is not set
        status = input0->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << input0->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        const int64_t trip_count = 0;
        status = input0->CopyFromHost(&trip_count);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "get trip count failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    auto input1 = subgraph->GetInputTensorImpl(1);
    if (input1->GetDevice()) { // not used by anyone if device is not set
        status = input1->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << input1->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        status = input1->CopyFromHost(&keep_going);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "get `keep_going` failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    for (uint32_t i = 2; i < ctx.GetInputCount(); ++i) {
        auto src = ctx.GetInput<TensorImpl>(i);
        auto dst = subgraph->GetInputTensorImpl(i);

        *dst->GetShape() = *src->GetShape();

        if (dst->GetDevice() == src->GetDevice()) {
            dst->SetBuffer(src->GetBufferDesc());
        } else {
            status = utils::CopyTensorBuffer(*src, dst, tmp_cpu_device);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy tensor from [" << src->GetName() << "] to [" << dst->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    for (uint32_t i = 0; i < subgraph->GetExtraInputCount(); ++i) {
        auto dst = subgraph->GetExtraInputTensorImpl(i);
        auto src = ctx.GetExtraInput<TensorImpl>(i);
        if (!src) {
            LOG(ERROR) << "cannot find extra input[" << dst->GetName() << "] from node extra inputs.";
            return RC_NOT_FOUND;
        }

        *dst->GetShape() = *src->GetShape();

        if (dst->GetDevice() == src->GetDevice()) {
            dst->SetBuffer(src->GetBufferDesc());
        } else {
            status = utils::CopyTensorBuffer(*src, dst, tmp_cpu_device);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "copy tensor from [" << src->GetName() << "] to [" << dst->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode SetOutputsFromInputs(const LoopInfo& info, const string& loop_kernel_name, Device* kernel_dev,
                                    Device* tmp_cpu_device, RuntimeImpl* subgraph, KernelExecContext* ctx) {
    // copy loop carried deps from loop's input
    for (uint32_t i = 0; i < info.loop_carried_dep_num; ++i) {
        auto src = ctx->GetInput<TensorImpl>(i + 2); // skip `M` and `cond`
        auto dst = ctx->GetOutput<TensorImpl>(i);

        dst->SetDevice(kernel_dev);
        *dst->GetShape() = *src->GetShape();

        auto status = utils::CopyTensorBuffer(*src, dst, tmp_cpu_device);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy from tensor[" << src->GetName() << "] to tensor[" << dst->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    // create empty scan outputs
    for (uint32_t i = info.loop_carried_dep_num; i < ctx->GetOutputCount(); ++i) {
        auto src = subgraph->GetOutputTensorImpl(i + 1); // skip `cond`
        auto& src_shape = *src->GetShape();

        if (src_shape.IsEmpty()) {
            LOG(WARNING) << "loop kernel[" << loop_kernel_name << "] trip count is 0 and "
                         << "cannot find output shape from subgraph.";
        } else {
            auto output = ctx->GetOutput<TensorImpl>(i);
            *output->GetShape() = src_shape;

            auto status = output->ReallocBuffer();
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "ReallocBuffer for empty scan output[" << output->GetName()
                           << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    return RC_SUCCESS;
}

static RetCode SetOutputsFromSubgraph(const LoopInfo& info, Device* kernel_dev, Device* tmp_cpu_device,
                                      LoopConcatOutputFunc concat_output_func, RuntimeImpl* subgraph,
                                      KernelExecContext* ctx) {
    // copy loop carried deps from subgraph's output
    for (uint32_t i = 0; i < info.loop_carried_dep_num; ++i) {
        auto src = subgraph->GetOutputTensorImpl(i + 1);
        auto dst = ctx->GetOutput<TensorImpl>(i);

        dst->SetDevice(kernel_dev);
        *dst->GetShape() = *src->GetShape();

        auto status = utils::CopyTensorBuffer(*src, dst, tmp_cpu_device);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy from tensor[" << src->GetName() << "] to tensor[" << dst->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    for (uint32_t i = info.loop_carried_dep_num; i < ctx->GetOutputCount(); ++i) {
        auto dst = ctx->GetOutput<TensorImpl>(i);

        auto& outputs = info.intermediate_outputs[i];
        auto& output_shape = *outputs[i].GetShape();

        vector<int64_t> dims(1 + output_shape.GetDimCount());
        dims[0] = outputs.size();
        for (uint32_t j = 0; j < output_shape.GetDimCount(); ++j) {
            dims[j + 1] = output_shape.GetDim(j);
        }

        auto dst_shape = dst->GetShape();
        dst_shape->SetDataType(output_shape.GetDataType());
        dst_shape->SetDataFormat(output_shape.GetDataFormat());
        dst_shape->Reshape(dims.data(), dims.size());

        auto status = dst->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << dst->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }

        status = concat_output_func(outputs, &dst->GetBufferDesc());
        if (status != RC_SUCCESS) {
            return status;
        }
    }

    return RC_SUCCESS;
}

RetCode LoopKernel::DoExecute(KernelExecContext* ctx) {
    LoopInfo loop_info(*ctx);
    utils::GenericCpuDevice tmp_cpu_device;

    bool keep_going;
    auto status = GetKeepGoing(*ctx, &keep_going);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "get `keep_going` failed: " << GetRetCodeStr(status);
        return status;
    }

    auto device = GetDevice();

    if (!keep_going) {
        return SetOutputsFromInputs(loop_info, GetName(), device, &tmp_cpu_device, &subgraph_, ctx);
    }

    const int64_t max_trip_count = GetMaxTripCount(*ctx);

    status = InitSubgraphInputs(*ctx, keep_going, &tmp_cpu_device, &subgraph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitSubgraphInputs of loop kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    int64_t trip_count = 0;
    while (trip_count < max_trip_count && keep_going) {
        if (trip_count != 0) {
            status = UpdateSubgraphInputs(trip_count, &subgraph_, &tmp_cpu_device);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "UpdateSubgraphInputs failed: " << GetRetCodeStr(status);
                return status;
            }
            status = SaveSubgraphOutputs(&subgraph_, &loop_info);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "SaveSubgraphOutputs failed: " << GetRetCodeStr(status);
                return status;
            }
        }

        status = subgraph_.Run();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "loop kernel[" << GetName() << "] Run() failed: " << GetRetCodeStr(status);
            return status;
        }

        ++trip_count;
        status = subgraph_.GetOutputTensorImpl(0)->CopyToHost(&keep_going);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "set `keep_going` failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    if (trip_count == 0) {
        status = SetOutputsFromInputs(loop_info, GetName(), device, &tmp_cpu_device, &subgraph_, ctx);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "SetOutputsFromInputs of loop kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    } else {
        SaveSubgraphOutputs(&subgraph_, &loop_info);
        status = SetOutputsFromSubgraph(loop_info, device, &tmp_cpu_device, concat_output_func_, &subgraph_, ctx);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "SetOutputsFromSubgraph of loop kernel[" << GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
