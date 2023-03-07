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

#include "ppl/nn/oputils/onnx/reshape_einsum.h"
#include "ppl/nn/oputils/broadcast.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
#include <vector>
#include <string>
using namespace ppl::common;
using namespace std;

namespace ppl { namespace nn { namespace onnx {

RetCode ParseLabels(const string& s, vector<vector<char>>& labels, int64_t num_ops){
    bool found_ell = false;
    int64_t curr_op = 0;
    for(uint64_t i=0;i<s.length();++i){
        switch(s[i]){
            case ' ':
                break;
            case '.':
                if(found_ell){
                    LOG(ERROR)<<"alread found one ellipsis." ;
                    return RC_INVALID_VALUE;
                }
                if(!found_ell && i+2<s.length() && s[++i]=='.'&&s[++i]=='.'){ //find valid ellipsis
                    labels[curr_op].push_back('.');
                    found_ell = true;
                    break;
                }else{
                    LOG(ERROR)<<"Invalid ellipsis found";
                    return RC_INVALID_VALUE;
                }
            case ',': // move to next operand
                ++curr_op;
                if(curr_op >= num_ops){
                    LOG(ERROR)<<"fewer operands were provided than specified in the equation";
                }
                found_ell = false;
                break;
            default: // parse label
                if(s[i]<'a' || s[i]>'z'){
                    LOG(ERROR)<<"Invalid ellipsis found";
                    return RC_INVALID_VALUE;
                }
                labels[curr_op].push_back(s[i]);
        }
    }
    return RC_SUCCESS;
}

//get unsqueeze shape and permute shape
// RetCode GetPermShape(vector<char>& perm_label, vector<char>& labels, vector<int>& perm_shape, int& usq_t){
//     return RC_SUCCESS;
// }

// replace ellipsis in the left labels with read dims
RetCode ReplaceEllipsis(vector<vector<char>>& labels, InputOutputInfo* info){
    for(uint64_t i=0;i<labels.size();++i){
        auto& label = labels[i];
        if(label[0] != '.')
            continue;
        auto shape = info->GetInput<TensorImpl>(i)->GetShape();
        auto ndim = shape->GetDimCount();
        int64_t ell_dim = ndim - (label.size() - 1);
        for(int64_t j = ell_dim-1;j >= 0;j--)
            label.insert(label.begin() + 1, shape->GetDim(j) + '0');
        label.erase(label.begin());
    }
    return RC_SUCCESS;
}
RetCode BroadCastCheck(vector<vector<int64_t>>& ell_map, vector<int64_t>& ell_shape, uint64_t max_ell_dim){
    // add 1 in front
    for(uint64_t i=0;i<ell_map.size();++i){
        for(uint64_t j=0;j<max_ell_dim - ell_map[i].size(); j++)
            ell_map[i].insert(ell_map[i].begin(), 1);
    }

    for(uint64_t i=0;i<max_ell_dim;++i){
        for(uint64_t j=0;j<ell_map.size();++j)
            ell_shape[i] = max(ell_shape[i], ell_map[j][i]);
    }
    // braodcast check
    for(uint64_t i=0;i<max_ell_dim;++i){
        for(uint64_t j=0;j<ell_map.size();++j)
            if(ell_map[j][i] !=1 && ell_map[j][i] != ell_shape[i])
                return RC_INVALID_VALUE;
    }
    return RC_SUCCESS;
}
RetCode ReshapeEinSum(InputOutputInfo* info, const ir::Attr* arg) {
    auto param = static_cast<const EinSumParam*>(arg);
    auto equation = param->equation;
    // parse the equation
    auto arrow_pos = equation.find("->");
    const auto num_ops = info->GetInputCount();

    auto lhs = equation.substr(0, arrow_pos);

    vector<vector<char>> lhs_labels(num_ops);
    auto status = ParseLabels(lhs, lhs_labels, num_ops);
    if(status != RC_SUCCESS){
        LOG(ERROR) <<"Parse labels <<[" << lhs << "] false";
        return status;
    };
    // Replace Ellipsis
    status = ReplaceEllipsis(lhs_labels, info);
    if(status != RC_SUCCESS){
        LOG(ERROR) <<"Deal ellipsis false";
    };

    // get output shape
    constexpr int TOTAL_LABELS = 'z' - 'a' + 1;
    vector<int> label_map(TOTAL_LABELS, -1);   // label to real dim
    vector<vector<int64_t>> ell_map(num_ops); // ellipsis to real dim
    for(uint64_t curr_op=0; curr_op < num_ops; ++curr_op){
        auto labels = lhs_labels[curr_op];
        auto shape = info->GetInput<TensorImpl>(curr_op)->GetShape();
        for(uint64_t j=0; j < labels.size(); ++j){
            auto label = labels[j];
            if('0' <= label && label <='9'){
                ell_map[curr_op].insert(ell_map[curr_op].begin(), label -'0');
                continue;
            }
            if(label_map[label - 'a'] != -1 && shape->GetDim(j) != label_map[label - 'a'] ){
                LOG(ERROR) << "input shape dim ["<< shape->GetDim(j) << "] didn't match label [" << label << "]";
                return RC_INVALID_VALUE;
            }
            if(label_map[label - 'a'] == -1)
                label_map[label-'a'] = shape->GetDim(j);
        }
    }
    // broadcast the dim represented by ellipsis
    uint64_t max_ell_dim = 0;
    for(uint64_t i=0;i<ell_map.size();++i)
        max_ell_dim = max(max_ell_dim, ell_map[i].size());
    vector<int64_t> ell_shape(max_ell_dim, 0);
    if(BroadCastCheck(ell_map, ell_shape, max_ell_dim) != RC_SUCCESS){
        LOG(ERROR)<<"Broadcast error";
    }

    vector<int64_t> output_shape;
    if(arrow_pos != string::npos){
        vector<vector<char>> rhs_labels(1);
        auto rhs = equation.substr(arrow_pos+2);
        ParseLabels(rhs, rhs_labels, 1);
        if(rhs_labels[0][0] == '.')
            output_shape.assign(ell_shape.begin(), ell_shape.end());
        for(uint64_t i=0;i<rhs_labels[0].size();++i){
            if(rhs_labels[0][i] != '.')
                output_shape.push_back(label_map[rhs_labels[0][i] - 'a']);
        }
    } else{ // when there is no right equaition, use ell_shape directly;
        if(ell_shape.size() == 0)
            output_shape.push_back(1);
        else
            output_shape.assign(ell_shape.begin(), ell_shape.end());
    }

    if (output_shape.size()==1 && output_shape[0]==1) {
        info->GetOutput<TensorImpl>(0)->GetShape()->ReshapeAsScalar();
    } else {
        info->GetOutput<TensorImpl>(0)->GetShape()->Reshape(output_shape);
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::onnx
