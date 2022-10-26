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

#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/pmx/ppl_shape_operation.h"
#include <vector>
using namespace flatbuffers;

namespace ppl { namespace nn { namespace pmx { namespace pmx {


Offset<ShapeOperationParam> SerializeShapeOperationParam(const std::vector<edgeid_t>& eid2seq, const ppl::nn::pmx::ShapeOperationParam& param, FlatBufferBuilder* builder) {
    std::vector<int32_t> edge_ids;
    std::vector<int32_t> numerator;
    std::vector<int32_t> denominator;
    std::vector<int32_t> real_dim;
    std::vector<bool> scalar;
    for (auto it = param.alpha.begin(); it != param.alpha.end(); it++) {
        int32_t edge_id = it->first < eid2seq.size() ? eid2seq[it->first] : -1;
        edge_ids.push_back(edge_id);
        for (int j = 0; j < (ShapeMatrix::MAXDIMSIZE + 1); ++j) {
            for (int k = 0; k < (ShapeMatrix::MAXDIMSIZE + 1); ++k) {
                numerator.push_back(it->second.numerator[j][k]);
            }
        }
        for (int j = 0; j < (ShapeMatrix::MAXDIMSIZE + 1); ++j) {
            for (int k = 0; k < (ShapeMatrix::MAXDIMSIZE + 1); ++k) {
                denominator.push_back(it->second.denominator[j][k]);
            }
        }
        real_dim.push_back(it->second.real_dim);
        scalar.push_back(it->second.scalar);
    }

    auto fb_edge_ids = builder->CreateVector(edge_ids);
    auto fb_numerator = builder->CreateVector(numerator);
    auto fb_denominator = builder->CreateVector(denominator);
    auto fb_real_dim = builder->CreateVector(real_dim);
    auto fb_scalar = builder->CreateVector(scalar);

    return CreateShapeOperationParam(*builder, fb_edge_ids, fb_numerator, fb_denominator, fb_real_dim, fb_scalar);
}

void DeserializeShapeOperationParam(const ShapeOperationParam& fb_param, ppl::nn::pmx::ShapeOperationParam* param) {
    std::vector<int32_t> edge_ids;
    std::vector<int32_t> numerator;
    std::vector<int32_t> denominator;
    std::vector<int32_t> real_dim;
    std::vector<bool> scalar;
    utils::Fbvec2Stdvec(fb_param.edge_ids(), &edge_ids);
    utils::Fbvec2Stdvec(fb_param.numerator(), &numerator);
    utils::Fbvec2Stdvec(fb_param.denominator(), &denominator);
    utils::Fbvec2Stdvec(fb_param.real_dim(), &real_dim);
    utils::Fbvec2Stdvec(fb_param.scalar(), &scalar);
    int count = edge_ids.size();
    int width = ShapeMatrix::MAXDIMSIZE + 1;
    for (int i = 0; i < count; ++i) {
        ShapeMatrix shape_matrix;
        for (int j = 0; j < (ShapeMatrix::MAXDIMSIZE + 1); ++j) {
            for (int k = 0; k < (ShapeMatrix::MAXDIMSIZE + 1); ++k) {
                int index = i * width * width + j * width + k;
                shape_matrix.numerator[j][k] = numerator[index];
            }
        }
        for (int j = 0; j < (ShapeMatrix::MAXDIMSIZE + 1); ++j) {
            for (int k = 0; k < (ShapeMatrix::MAXDIMSIZE + 1); ++k) {
                int index = i * width * width + j * width + k;
                shape_matrix.denominator[j][k] = denominator[index];
            }
        }
        shape_matrix.real_dim = real_dim[i];
        shape_matrix.scalar = scalar[i];
        param->alpha[edge_ids[i]] = shape_matrix;
    }
}

}}}} // namespace ppl::nn::pmx::onnx
