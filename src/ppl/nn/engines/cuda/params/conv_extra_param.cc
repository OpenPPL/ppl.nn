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

#include "ppl/nn/engines/cuda/params/conv_extra_param.h"

#include "ppl/nn/params/onnx/leaky_relu_param.h"
#include "ppl/nn/common/logger.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include <fstream>
#include <sstream>

using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

int GetRelueType(const std::string& name) {
    if (name == "Relu")
        return 0;
    if (name == "Clip")
        return 1;
    if (name == "PRelu")
        return 2;
    if (name == "LeakyRelu")
        return 3;
    if (name == "Sigmoid")
        return 4;
    return -1;
}

#define GetPadSize(pad_size, type){ \
    pad_size = 0; \
    if (type == DATATYPE_FLOAT32) { \
            pad_size = 4; \
    } else if (type == DATATYPE_FLOAT16) { \
            pad_size = 8; \
    } else if (type == DATATYPE_INT8) { \
            pad_size = 16; \
    } \
}

#define Align(x, y) ( ((x)+(y)-1) / (y) * (y) )

RetCode ConvertToForwardConvParam(const TensorShape& shape_in0, const TensorShape& shape_in1,
                                  const TensorShape& shape_out, ConvolutionParam normal_param,
                                  conv_param_t& conv_param) {
    conv_param.in_height = shape_in0.GetDim(2);
    conv_param.in_width = shape_in0.GetDim(3);
    conv_param.in_num = shape_in0.GetDim(0);
    conv_param.num_grp = normal_param.group;
    conv_param.num_chl = shape_in1.GetDim(1) * normal_param.group;
    conv_param.num_flt = shape_in1.GetDim(0);
    unsigned int in_pad_size;
    unsigned int flt_pad_size;
    GetPadSize(in_pad_size, shape_in0.GetDataType());
    GetPadSize(flt_pad_size, shape_in1.GetDataType());
    //conv_param.num_chl_pad = (conv_param.num_chl + 7) / 8 * 8;
    //conv_param.num_flt_pad = (conv_param.num_flt + 7) / 8 * 8;
    //std::cout << "in pad size: " << in_pad_size << " flt_pad_size: " << flt_pad_size << std::endl;
    conv_param.num_chl_pad = Align(conv_param.num_chl, in_pad_size);
    conv_param.num_flt_pad = Align(conv_param.num_flt, flt_pad_size);
    conv_param.flt_height = shape_in1.GetDim(2);
    conv_param.flt_width = shape_in1.GetDim(3);
    conv_param.out_height = shape_out.GetDim(2);
    conv_param.out_width = shape_out.GetDim(3);
    conv_param.stride_height = normal_param.strides[0];
    conv_param.stride_width = normal_param.strides[1];
    conv_param.pad_height = normal_param.pads[0];
    conv_param.pad_width = normal_param.pads[1];
    conv_param.hole_height = normal_param.dilations[0];
    conv_param.hole_width = normal_param.dilations[1];
    conv_param.has_bias = normal_param.bias_term;
    return RC_SUCCESS;
}
#undef GetPadSize
#undef Align

RetCode ConvertToPrelu(uint32_t fuse_index, InputOutputInfo* info, CudaDevice* device, ConvFusionInfo fuse_info,
                       fuse_param_t& fuse_param) {
    uint32_t prelu_input = fuse_info.input_ind[fuse_index];
    auto shape = info->GetInput<TensorImpl>(prelu_input)->GetShape();

    if (fuse_index == 0) {
        fuse_param.has_prelu = shape.IsScalar() ? 1 : 2;
        fuse_param.prelu = info->GetInput<TensorImpl>(prelu_input)->GetBufferPtr();
    } else {
        fuse_param.has_elt_prelu = shape.IsScalar() ? 1 : 2;
        fuse_param.elt_prelu = info->GetInput<TensorImpl>(prelu_input)->GetBufferPtr();
    }

    return RC_SUCCESS;
}

RetCode ConvertToLeakyrelu(uint32_t fuse_index, InputOutputInfo* info, CudaDevice* device, ConvFusionInfo fuse_info,
                           fuse_param_t& fuse_param) {
    if (fuse_index == 0) {
        fuse_param.has_prelu = 1;
        fuse_param.leaky = ((LeakyReLUParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
    } else {
        fuse_param.has_elt_prelu = 1;
        fuse_param.elt_leaky = ((LeakyReLUParam*)fuse_info.fuse_attrs[fuse_index])->alpha;
    }

    return RC_SUCCESS;
}

RetCode ConvertToForwardFuseParam(InputOutputInfo* info, CudaDevice* device, ConvFusionInfo fuse_info,
                                  fuse_param_t& fuse_param) {
    const std::set<std::string> relu_set{"Relu", "Clip", "PRelu", "LeakyRelu", "Sigmoid"};
    int fuse_index = 0;
    int fuse_size = fuse_info.types.size();

    RetCode status;
    ClipParam* param;

    if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType(fuse_info.types[fuse_index]);
        switch (type) {
            case 0: // Relu
                fuse_param.has_activation = 1;
                break;
            case 1: // Clip
                fuse_param.has_clip = true;
                param = (ClipParam*)fuse_info.fuse_attrs[fuse_index];
                fuse_param.clip_min = param->min_val;
                fuse_param.clip_max = param->max_val;
                break;
            case 2: // PRelu
                status = ConvertToPrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 3: // LeakyRelu
                status = ConvertToLeakyrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 4: // Sigmoid
                fuse_param.has_activation = 2;
                break;
            default:
                return RC_UNSUPPORTED;
        }
        fuse_index++;
    }

    if (fuse_index < fuse_size && fuse_info.types[fuse_index] == "Add") {
        fuse_param.has_elt = true;
        uint32_t elt_input = fuse_info.input_ind[fuse_index];
        fuse_param.pre_data = info->GetInput<TensorImpl>(elt_input)->GetBufferPtr();
        fuse_index++;
    }

    if (fuse_index < fuse_size && fuse_param.has_elt && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        int type = GetRelueType(fuse_info.types[fuse_index]);
        switch (type) {
            case 0: // Relu
                fuse_param.has_elt_activation = 1;
                break;
            case 1: // Clip
                fuse_param.has_elt_clip = true;
                param = (ClipParam*)fuse_info.fuse_attrs[fuse_index];
                fuse_param.elt_clip_min = param->min_val;
                fuse_param.elt_clip_max = param->max_val;
                break;
            case 2: // PRelu
                status = ConvertToPrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 3: // LeakyRelu
                status = ConvertToLeakyrelu(fuse_index, info, device, fuse_info, fuse_param);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Set prelu fuse info failed: " << GetRetCodeStr(status);
                    return status;
                }
                break;
            case 4: // Sigmoid
                fuse_param.has_elt_activation = 2;
                break;
            default:
                return RC_UNSUPPORTED;
        }
    }

    fuse_param.has_concat = fuse_info.channel_offset >= 0;
    if (fuse_param.has_concat) {
        fuse_param.concat_offset = fuse_info.channel_offset;
        fuse_param.concat_stride = fuse_info.channel_size;
        fuse_param.post_concat = info->GetOutput<TensorImpl>(0)->GetBufferPtr();
    }

    return RC_SUCCESS;
}

void LoadAlgoInfo(const std::string& file_path, const algo_param_t& algo_param, const std::string& key_str) {
    if (!file_path.empty()) {
        rapidjson::Document oWriteDoc;
        fstream ifile;
        ifile.open(file_path, ios_base::in);
        if (!ifile.is_open()) {
            oWriteDoc.SetObject();
        } else {
            std::stringstream algo_info_json_buffer;
            algo_info_json_buffer << ifile.rdbuf();
            oWriteDoc.Parse(algo_info_json_buffer.str().c_str());
            if (oWriteDoc.HasParseError()) {
                oWriteDoc.SetObject();
            }
        }
        ifile.close();
        
        std::string kname_str = algo_param.algo_name;

        rapidjson::Document::AllocatorType& allocator = oWriteDoc.GetAllocator();
        rapidjson::Value object(rapidjson::kObjectType);
        rapidjson::Value key_info(key_str.c_str(), key_str.size(), allocator);
        rapidjson::Value kname_info(kname_str.c_str(), kname_str.size(), allocator);
        object.AddMember("kid", algo_param.kid, allocator);
        object.AddMember("kname", kname_info, allocator);
        object.AddMember("splitk", algo_param.splitk, allocator);
        object.AddMember("splitf", algo_param.splitf, allocator);
        oWriteDoc.RemoveMember(key_info);
        oWriteDoc.AddMember(key_info, object, allocator);
        rapidjson::StringBuffer oBuffer;
        rapidjson::Writer<rapidjson::StringBuffer> oWriter(oBuffer);
        oWriteDoc.Accept(oWriter);

        fstream ofile;
        ofile.open(file_path, ios_base::out);
        if (!ofile.is_open()) {
            return;
        }
        ofile << oBuffer.GetString();
        ofile.close();
    }
    return;
}

}}} // namespace ppl::nn::cuda
