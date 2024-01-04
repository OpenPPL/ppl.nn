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

#include "slice_op.h"

#include "ppl/nn/engines/llm_cuda/kernels/onnx/slice_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_slice.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/llm_cuda/pmx/generated/llm_cuda_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;


namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {

RetCode SliceOp::CommonInit() {
    infer_type_and_format_func_ = GenericInferTypeAndFormat;
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        std::vector<int64_t> local_starts;
        std::vector<int64_t> local_ends;
        std::vector<int64_t> local_axes;
        std::vector<int64_t> local_steps;

        int64_t *starts_data = nullptr;
        int64_t *ends_data = nullptr;
        int64_t *axes_data = nullptr;
        int64_t *steps_data = nullptr;
        int64_t num_axes = 0;

        auto starts = info->GetInput<TensorImpl>(1);
        if (!starts->GetBufferPtr()) {
            return RC_NOT_FOUND;
        }
        num_axes = starts->GetShape()->CalcElementsIncludingPadding();

        if (constant_slice_param_.starts.empty() && num_axes > 0) {
            local_starts.resize(num_axes);
            auto status = starts->CopyToHost(local_starts.data());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "starts->CopyToHost() failed: " << GetRetCodeStr(status);
                return status;
            }
            starts_data = local_starts.data();
        } else {
            starts_data = constant_slice_param_.starts.data();
        }

        if (constant_slice_param_.ends.empty() && num_axes > 0) {
            auto ends = info->GetInput<TensorImpl>(2);
            if (!ends->GetBufferPtr()) {
                return RC_NOT_FOUND;
            }

            local_ends.resize(num_axes);
            auto status = ends->CopyToHost(local_ends.data());
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "ends->CopyToHost() failed: " << GetRetCodeStr(status);
                return status;
            }
            ends_data = local_ends.data();
        } else {
            ends_data = constant_slice_param_.ends.data();
        }

        if (constant_slice_param_.axes.empty() && num_axes > 0) {
            if (info->GetInputCount() > 3) {
                auto axes = info->GetInput<TensorImpl>(3);
                if (!axes->GetBufferPtr()) {
                    return RC_NOT_FOUND;
                }

                local_axes.resize(num_axes);
                auto status = axes->CopyToHost(local_axes.data());
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "axes->CopyToHost() failed: " << GetRetCodeStr(status);
                    return status;
                }
            } else {
                local_axes.resize(num_axes);
                for (int64_t i = 0; i < num_axes; ++i) {
                    local_axes[i] = i;
                }
            }
            axes_data = local_axes.data();
        } else {
            axes_data = constant_slice_param_.axes.data();
        }

        if (constant_slice_param_.steps.empty() && num_axes > 0) {
            if (info->GetInputCount() > 4) {
                auto steps = info->GetInput<TensorImpl>(4);
                if (!steps->GetBufferPtr()) {
                    return RC_NOT_FOUND;
                }

                local_steps.resize(num_axes);
                auto status = steps->CopyToHost(local_steps.data());
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "steps->CopyToHost() failed: " << GetRetCodeStr(status);
                    return status;
                }
            } else {
                local_steps.assign(num_axes, 1);
            }
            steps_data = local_steps.data();
        } else {
            steps_data = constant_slice_param_.steps.data();
        }

        return ppl::nn::onnx::ReshapeSlice(
            info, starts_data, ends_data, axes_data, steps_data, num_axes);
    };

    return RC_SUCCESS;
}

RetCode SliceOp::DoInit(const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph->data;

    {
        auto starts_data_it = graph_data->constants.find(node->GetInput(1));
        const int64_t* starts_data = nullptr;
        if (starts_data_it != graph_data->constants.end()) {
            starts_data = (const int64_t*)starts_data_it->second.data.GetData();
        }

        if (starts_data != nullptr) {
            auto starts_shape_it = graph_data->shapes.find(node->GetInput(1));
            if (starts_shape_it != graph_data->shapes.end()) {
                auto& starts_shape = starts_shape_it->second;
                constant_slice_param_.starts.assign(starts_data, starts_data + starts_shape.dims[0]);
            }
        } else {
            LOG(WARNING) << "non-constant slice starts will cause performance downgrade.";
        }
    }

    {
        auto ends_data_it = graph_data->constants.find(node->GetInput(2));
        const int64_t* ends_data = nullptr;
        if (ends_data_it != graph_data->constants.end()) {
            ends_data = (const int64_t*)ends_data_it->second.data.GetData();
        }

        if (ends_data != nullptr) {
            auto ends_shape_it = graph_data->shapes.find(node->GetInput(2));
            if (ends_shape_it != graph_data->shapes.end()) {
                auto& ends_shape = ends_shape_it->second;
                constant_slice_param_.ends.assign(ends_data, ends_data + ends_shape.dims[0]);
            }
        } else {
            LOG(WARNING) << "non-constant slice ends will cause performance downgrade.";
        }
    }

    if (node->GetInputCount() > 3) {
        auto axes_data_it = graph_data->constants.find(node->GetInput(3));
        const int64_t* axes_data = nullptr;
        if (axes_data_it != graph_data->constants.end()) {
            axes_data = (const int64_t*)axes_data_it->second.data.GetData();
        }

        if (axes_data != nullptr) {
            auto axes_shape_it = graph_data->shapes.find(node->GetInput(3));
            if (axes_shape_it != graph_data->shapes.end()) {
                auto& axes_shape = axes_shape_it->second;
                constant_slice_param_.axes.assign(axes_data, axes_data + axes_shape.dims[0]);
            }
        } else {
            LOG(WARNING) << "non-constant slice axes will cause performance downgrade.";
        }
    } else {
        constant_slice_param_.axes.resize(constant_slice_param_.starts.size());
        for (uint64_t i = 0; i < constant_slice_param_.axes.size(); ++i) {
            constant_slice_param_.axes[i] = i;
        }
    }

    if (node->GetInputCount() > 4) {
        auto steps_data_it = graph_data->constants.find(node->GetInput(4));
        const int64_t* steps_data = nullptr;
        if (steps_data_it != graph_data->constants.end()) {
            steps_data = (const int64_t*)steps_data_it->second.data.GetData();
        }

        if (steps_data != nullptr) {
            auto steps_shape_it = graph_data->shapes.find(node->GetInput(4));
            if (steps_shape_it != graph_data->shapes.end()) {
                auto& steps_shape = steps_shape_it->second;
                constant_slice_param_.steps.assign(steps_data, steps_data + steps_shape.dims[0]);
            }
        } else {
            LOG(WARNING) << "non-constant slice steps will cause performance downgrade.";
        }
    } else {
        constant_slice_param_.steps.assign(constant_slice_param_.starts.size(), 1);
    }

    return CommonInit();
}

KernelImpl* SliceOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<SliceKernel>(&constant_slice_param_);
}

#ifdef PPLNN_ENABLE_PMX_MODEL
ppl::common::RetCode SliceOp::SerializeData(const ppl::nn::pmx::SerializationContext&, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder builder;
    
    auto fb_starts = builder.CreateVector(constant_slice_param_.starts);
    auto fb_ends   = builder.CreateVector(constant_slice_param_.ends);
    auto fb_axes   = builder.CreateVector(constant_slice_param_.axes);
    auto fb_steps  = builder.CreateVector(constant_slice_param_.steps);
    
    auto fb_param = pmx::CreateSliceParam(builder, fb_starts, fb_ends, fb_axes, fb_steps);
    auto fb_op_param = pmx::CreateOpParam(builder, pmx::OpParamType_SliceParam, fb_param.Union());
    pmx::FinishOpParamBuffer(builder, fb_op_param);
    return ds->Write(builder.GetBufferPointer(), builder.GetSize());
}

ppl::common::RetCode SliceOp::DeserializeData(const ppl::nn::pmx::DeserializationContext&, const void* base, uint64_t size) {
    auto fb_op_param = pmx::GetOpParam(base);
    auto fb_param = fb_op_param->value_as_SliceParam();
    
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_param->starts(), &constant_slice_param_.starts);
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_param->ends(),   &constant_slice_param_.ends);
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_param->axes(),   &constant_slice_param_.axes);
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_param->steps(),  &constant_slice_param_.steps);

    return CommonInit();
}
#endif

}}}}} // namespace ppl::nn::llm::cuda::pmx
