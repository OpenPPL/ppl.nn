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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_PMX_UTILS_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_PMX_UTILS_H_

#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/cuda/params/conv_extra_param.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"
#include "ppl/nn/engines/cuda/params/convtranspose_extra_param.h"
#include "ppl/nn/engines/cuda/pmx/generated/cuda_op_params_generated.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv.h"

using namespace ppl::nn;
using namespace ppl::nn::cuda;

namespace ppl { namespace nn { namespace pmx { namespace cuda {

template <typename T>
static RetCode SerializePrivateData(const pmx::SerializationContext& ctx, const T& extra_param, const std::string& ptx_code, flatbuffers::FlatBufferBuilder* builder) {
    auto& tiles = extra_param.algo_info.tiles;
    vector<int32_t> tiles_vec{tiles.m_cta,
                              tiles.n_cta,
                              tiles.k_cta,
                              tiles.m_warp,
                              tiles.n_warp,
                              tiles.k_per_step,
                              tiles.k_per_set,
                              tiles.flt_size,
                              tiles.flt_pad_size,
                              tiles.cta_size_in_thd,
                              tiles.smem_size,
                              tiles.buf};
    auto fb_algo_info = pmx::cuda::CreateConvAlgoInfoDirect(*builder,
                                                            extra_param.algo_info.algo_type.c_str(),
                                                            extra_param.algo_info.algo_name.c_str(),
                                                            extra_param.algo_info.conv_type.c_str(),
                                                            extra_param.algo_info.mma_shape.c_str(),
                                                            &tiles_vec,
                                                            extra_param.algo_info.kid,
                                                            extra_param.algo_info.splitk,
                                                            extra_param.algo_info.splitf,
                                                            extra_param.is_initializer_weight,
                                                            extra_param.bias_term);

    std::vector<flatbuffers::Offset<flatbuffers::String>> fb_types;
    std::vector<pmx::cuda::FuseAttrs> fb_attrs;
    for (uint32_t i = 0; i < extra_param.fuse_info.types.size(); ++i) {
        auto type = extra_param.fuse_info.types[i];
        auto fb_type = builder->CreateString(type);
        fb_types.push_back(fb_type);
        pmx::cuda::FuseAttrs fb_attr;
        if (type == "Clip") {
            auto attr = extra_param.fuse_info.fuse_attrs[i];
            auto clip_attr = (CudaClipParam*)attr;
            fb_attr = pmx::cuda::FuseAttrs(clip_attr->min_value, clip_attr->max_value, 0);
        } else if (type == "LeakyRelu") {
            auto attr = extra_param.fuse_info.fuse_attrs[i];
            auto leaky_attr = (LeakyReluParam*)attr;
            fb_attr = pmx::cuda::FuseAttrs(0, 0, leaky_attr->alpha);
        } else {
            fb_attr = pmx::cuda::FuseAttrs(0, 0, 0);
        }
        fb_attrs.push_back(fb_attr);
    }

    // edge_id has reset in sequence
    edgeid_t fb_concat_edge_it = -1;
    if (extra_param.fuse_info.concat_edge_id != -1) {
        fb_concat_edge_it = ctx.eid2seq[extra_param.fuse_info.concat_edge_id];
    }
    auto fb_fuse_info = pmx::cuda::CreateConvFusionInfoDirect(*builder,
                                                              &fb_types,
                                                              &extra_param.fuse_info.input_inds,
                                                              &fb_attrs,
                                                              extra_param.fuse_info.channel_size,
                                                              extra_param.fuse_info.channel_offset,
                                                              fb_concat_edge_it);

    auto fb_code = builder->CreateVector((const uint8_t*)ptx_code.data(), ptx_code.size());
    auto fb_conv_param = pmx::cuda::CreateConvParam(*builder, fb_algo_info, fb_fuse_info, fb_code);
    auto fb_op_param = pmx::cuda::CreateOpParam(*builder, pmx::cuda::OpParamType_ConvParam, fb_conv_param.Union());
    pmx::cuda::FinishOpParamBuffer(*builder, fb_op_param);
    return RC_SUCCESS;
}

template <typename T>
static RetCode DeserializePrivateData(const void* fb_param, uint64_t size, std::string& ptx_code, T* extra_param) {
    auto fb_op_param = pmx::cuda::GetOpParam(fb_param);
    auto fb_conv_param = fb_op_param->value_as_ConvParam();

    auto fb_algo_info = fb_conv_param->algo_info();
    extra_param->algo_info.algo_type = fb_algo_info->algo_type()->c_str();
    extra_param->algo_info.algo_name = fb_algo_info->algo_name()->c_str();
    extra_param->algo_info.conv_type = fb_algo_info->conv_type()->c_str();
    extra_param->algo_info.mma_shape = fb_algo_info->mma_shape()->c_str();
    extra_param->algo_info.kid = fb_algo_info->kid();
    extra_param->algo_info.splitk = fb_algo_info->splitk();
    extra_param->algo_info.splitf = fb_algo_info->splitf();
    extra_param->is_initializer_weight = fb_algo_info->is_initializer_weight();
    extra_param->bias_term = fb_algo_info->has_bias();
    auto& tiles = extra_param->algo_info.tiles;
    std::vector<int> tiles_vec;
    ppl::nn::pmx::utils::Fbvec2Stdvec(fb_algo_info->tiles(), &tiles_vec);
    tiles.m_cta = tiles_vec[0];
    tiles.n_cta = tiles_vec[1];
    tiles.k_cta = tiles_vec[2];
    tiles.m_warp = tiles_vec[3];
    tiles.n_warp = tiles_vec[4];
    tiles.k_per_step = tiles_vec[5];
    tiles.k_per_set = tiles_vec[6];
    tiles.flt_size = tiles_vec[7];
    tiles.flt_pad_size = tiles_vec[8];
    tiles.cta_size_in_thd = tiles_vec[9];
    tiles.smem_size = tiles_vec[10];
    tiles.buf = tiles_vec[11];

    auto fb_fuse_info = fb_conv_param->fuse_info();
    auto fb_types = fb_fuse_info->types();
    auto fb_input_inds = fb_fuse_info->input_inds();
    auto fb_fuse_attrs = fb_fuse_info->fuse_attrs();

    extra_param->fuse_info.types.clear();
    extra_param->fuse_info.input_inds.clear();
    extra_param->fuse_info.fuse_attrs.clear();

    for (uint32_t i = 0; i < fb_types->size(); ++i) {
        std::string str = fb_types->Get(i)->str();
        extra_param->fuse_info.types.push_back(str);
        auto ind = fb_input_inds->Get(i);
        extra_param->fuse_info.input_inds.push_back(ind);
        auto attr = fb_fuse_attrs->Get(i);
        if (str == "Clip") {
            CudaClipParam clip;
            clip.max_value = attr->clip_max();
            clip.min_value = attr->clip_min();
            extra_param->fuse_info.fuse_attrs.push_back(std::move(&clip));
        } else if (str == "LeakyRelu") {
            LeakyReluParam leaky;
            leaky.alpha = attr->leaky_alpha();
            extra_param->fuse_info.fuse_attrs.push_back(std::move(&leaky));
        } else {
            extra_param->fuse_info.fuse_attrs.push_back(nullptr);
        }
    }

    extra_param->fuse_info.channel_offset = fb_fuse_info->channel_offset();
    extra_param->fuse_info.channel_size = fb_fuse_info->channel_size();
    extra_param->fuse_info.concat_edge_id = fb_fuse_info->concat_edge_id();

    ptx_code.assign((const char*)fb_conv_param->gene_code()->data(), fb_conv_param->gene_code()->size());
    return RC_SUCCESS;
}

}}}} // namespace ppl::nn::pmx::cuda

#endif