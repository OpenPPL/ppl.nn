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

#include "cudakernel/nn/conv/gene_kernel.h"
#include "gene_header.h"
#include "ppl/nn/common/logger.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <set>

std::string IntToString(int v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

std::string GetSizeString(float size)
{
    if (size == 0.0625) {
        return "_16TH";
    } else if (size == 0.125) {
        return "_8TH";
    } else if (size == 0.25) {
        return "_QTR";
    } else if (size == 0.5) {
        return "_HALF";
    } else if (size == 1.0) {
        return "1";
    } else if (size == 2.0) {
        return "2";
    } else if (size == 4.0) {
        return "4";
    } else if (size == 8.0) {
        return "8";
    } else if (size == 16.0) {
        return "16";
    }
    return "";
}

void WriteIncludeFile(std::stringstream& file_str, std::string path)
{
#ifdef PPLNN_ENABLE_CUDA_JIT
    auto header_str = GeneHeader::Instance()->Find(path);
    file_str << header_str << "\n\n";
#endif
    return;
}

ppl::common::RetCode Gene2spkKernel(std::string& file_res, std::string& kname, int cta_y, int cta_x, int warp_y, int warp_x, int k_size, int s_size, int splitk, int splitf, int buf_size, int declare_times)
{
    int WARP_SIZE      = 32;
    int INT4_TO_4HALF2 = 8;
    int MMA_Y          = 16;
    int MMA_X          = 8;
    int MMA_Y_HALF     = MMA_Y / 2;

    int cta_num  = cta_y * cta_x / warp_y / warp_x;
    int cta_size = cta_num * k_size / s_size * WARP_SIZE;

    // int sAv4_size = warp_y / INT4_TO_4HALF2;
    // int sBv4_size = warp_x / INT4_TO_4HALF2;
    float dAv4_size = (cta_y * k_size * 1.0) / (INT4_TO_4HALF2 * cta_size);
    float dBv4_size = (cta_x * k_size * 1.0) / (INT4_TO_4HALF2 * cta_size);

    auto flt_size = kname.substr(25, 2);
    std::stringstream file_str;

    file_str << "#define TILE_N_PER_CTA       " << cta_x << "\n";
    file_str << "#define TILE_M_PER_CTA       " << cta_y << "\n\n";

    file_str << "#define TILE_N_PER_WARP      " << warp_x << "\n";
    file_str << "#define TILE_M_PER_WARP      " << warp_y << "\n\n";

    file_str << "#define TILE_K_PER_CTA       " << k_size << "\n";
    file_str << "#define TILE_K_PER_SET       " << s_size << "\n";
    file_str << "#define TILE_K_PER_WARP      " << s_size << "\n\n";

    file_str << "#define INTER_SET_REDUCE_RATIO  ((TILE_K_PER_CTA) / (TILE_K_PER_SET))\n\n";

    if (k_size / s_size == 2) {
        file_str << "#define REDUCE(_h2R)            REDUCE_HALF2_1x4(_h2R)\n\n";
        file_str << "#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)        READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read)\n\n";
    } else if (k_size / s_size == 4) {
        file_str << "#define REDUCE(_h2R)            REDUCE_HALF2_3x4(_h2R)\n\n";
        file_str << "#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)        READ_sRv4_SIZE4(_Rv4, _sm_base_v4, _sRv4_read)\n\n";
    } else if (k_size / s_size == 1) {
        if (k_size == 32 && s_size == 32) {
            file_str << "#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)        READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read)\n\n";
        } else {
            file_str << "#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)        READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read)\n\n";
        }
    } else {
        LOG(ERROR) << "knum is error, create kernel failed with ksize " << k_size << " and s_size " << s_size;
        return ppl::common::RC_INVALID_VALUE;
    }

    file_str << "#define KERNEL_NAME " << kname << "\n";

    file_str << "#define USE_" << buf_size << "BUF\n\n";

    file_str << "#include <cuda_fp16.h>\n\n";
    if (splitk == 1 && splitf == 1)
        file_str << "#define ENABLE_FUSE 1\n\n";
    if (splitk > 1)
        file_str << "#define ENABLE_SPLITK\n";
    if (splitf > 1) {
        file_str << "#define ENABLE_SPLITF\n";
    }

    file_str << "#define uint int\n\n";
    file_str << "#define uint32_t int\n\n";

    if (declare_times == 0) {
        file_str << "#define MAX_LUT_SIZE 128\n\n";
        file_str << "#define MAX_SPLITK_SIZE 8\n\n";
        file_str << "struct lut_t{ int idx[MAX_LUT_SIZE]; };\n\n";
        file_str << "struct chl_lut_t{ int idx[MAX_SPLITK_SIZE + 1]; };\n\n";
        file_str << "struct kloop_lut_t{ int idx[MAX_SPLITK_SIZE + 1]; };\n\n";
    }

    WriteIncludeFile(file_str, "/2spk/common/const_macros.h");
    WriteIncludeFile(file_str, "/2spk/" + flt_size + "/bound_macros.h");
    WriteIncludeFile(file_str, "/2spk/common/ldsm_macros.h");
    WriteIncludeFile(file_str, "/2spk/" + flt_size + "/dmem_macros.h");
    WriteIncludeFile(file_str, "/2spk/common/hmma_macros.h");
    WriteIncludeFile(file_str, "/2spk/common/reduce_macros.h");
    WriteIncludeFile(file_str, "/2spk/common/smem_macros.h");

    file_str << "#define MMA_INSTS(_C, _A, _B)           MMA_INST_" << warp_y / MMA_Y << "x" << warp_x / MMA_X << "(_C, _A, _B)\n\n";

    if (warp_y == 128) {
        file_str << "#define READ_sAv1(_A, _sm_base_v1, _sAv1_read)          READ_sUv1_4x" << warp_y / MMA_Y_HALF / 4 << "(_A, _sm_base_v1, _sAv1_read)\n";
    } else if (warp_y == 64) {
        file_str << "#define READ_sAv1(_A, _sm_base_v1, _sAv1_read)          READ_sUv1_2x" << warp_y / MMA_Y_HALF / 2 << "(_A, _sm_base_v1, _sAv1_read)\n";
    } else if (warp_y == 16 || warp_y == 32) {
        file_str << "#define READ_sAv1(_A, _sm_base_v1, _sAv1_read)          READ_sUv1_1x" << warp_y / MMA_Y_HALF << "(_A, _sm_base_v1, _sAv1_read)\n";
    } else {
        LOG(ERROR) << "warp_y is error, create kernel failed";
        return ppl::common::RC_INVALID_VALUE;
    }

    file_str << "#define READ_sBv1(_B, _sm_base_v1, _sBv1_read)          READ_sUv1_1x" << warp_x / MMA_X << "(_B, _sm_base_v1, _sBv1_read)\n\n";
    file_str << "#define WRITE_sRv1(_sm_base_v1, _sRv1_write_base, _C)   WRITE_sRv1_" << warp_y / MMA_Y_HALF << "x" << warp_x / MMA_X << "(_sm_base_v1, _sRv1_write_base, _C)\n\n";

    if (flt_size == "f1" || flt_size == "fs") {
        file_str << "#define LOAD_dAv4(_regA, _dA, _dAv4_off, _in_c_v8_id, _in_hw_valid)     LOAD_dAv4_SIZE" << GetSizeString(dAv4_size) << "(_regA, _dA, _dAv4_off, _in_c_v8_id, _in_hw_valid)\n";
        file_str << "#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE" << GetSizeString(dAv4_size) << "(_sm_base_v4, _sm_off, _reg)\n\n";

        file_str << "#define LOAD_dBv4(_regB, _dB, _dBv4_off, _flt_c_v8_id, _flt_n_valid)    LOAD_dBv4_SIZE" << GetSizeString(dBv4_size) << "(_regB, _dB, _dBv4_off, _flt_c_v8_id, _flt_n_valid)\n";
        file_str << "#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE" << GetSizeString(dBv4_size) << "(_sm_base_v4, _sm_off, _reg)\n\n";

        file_str << "#define FLT_SIZE1\n\n";

    } else if (flt_size == "f3") {
        file_str << "#define LOAD_dAv4(_regA, _dA, _dAv4_off, _in_c_v8_id, _flt_hw_bid)      LOAD_dAv4_SIZE" << GetSizeString(dAv4_size) << "(_regA, _dA, _dAv4_off, _in_c_v8_id, _flt_hw_bid)\n";
        file_str << "#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE" << GetSizeString(dAv4_size) << "(_sm_base_v4, _sm_off, _reg)\n\n";

        file_str << "#define LOAD_dBv4(_regB, _dB, _dBv4_off, _flt_c_v8_id, _flt_n_valid)    LOAD_dBv4_SIZE" << GetSizeString(dBv4_size) << "(_regB, _dB, _dBv4_off, _flt_c_v8_id, _flt_n_valid)\n";
        file_str << "#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE" << GetSizeString(dBv4_size) << "(_sm_base_v4, _sm_off, _reg)\n\n";

        file_str << "#define FLT_SIZE3\n\n";
    } else if (flt_size == "fn") {
        file_str << "#define LOAD_dAv4(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id)  LOAD_dAv4_SIZE" << GetSizeString(dAv4_size) << "(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id)\n";
        file_str << "#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE" << GetSizeString(dAv4_size) << "(_sm_base_v4, _sm_off, _reg)\n\n";

        file_str << "#define LOAD_dBv4(_regB, _dB, _dBv4_off, _flt_c_v8_id, _flt_n_valid)    LOAD_dBv4_SIZE" << GetSizeString(dBv4_size) << "(_regB, _dB, _dBv4_off, _flt_c_v8_id, _flt_n_valid)\n";
        file_str << "#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE" << GetSizeString(dBv4_size) << "(_sm_base_v4, _sm_off, _reg)\n\n";

        file_str << "#define FWD_FLT(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid)    FWD_FLT_SIZE" << (dAv4_size >= 1 ? GetSizeString(dAv4_size) : "1") << "(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid)\n";

        file_str << "#define FLT_SIZEN\n\n";
    }

    WriteIncludeFile(file_str, "/2spk/common/output_macros.h");
    file_str << "extern \"C\" {\n\n";
    WriteIncludeFile(file_str, "/2spk/common/main_body.h");
    file_str << "}\n\n";
    WriteIncludeFile(file_str, "/2spk/common/uni_undefs.h");
    if (splitk == 1 && splitf == 1)
        file_str << "#undef ENABLE_FUSE\n\n";
    if (splitk > 1)
        file_str << "#undef ENABLE_SPLITK\n";
    if (splitf > 1) {
        file_str << "#undef ENABLE_SPLITF\n";
    }
    file_res = file_str.str();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode GeneIdxnKernel(std::string& file_res, std::string& kname, int cta_y, int cta_x, int warp_y, int warp_x, int k_size, int s_size, int declare_times)
{
    // iggnt WARP_SIZE = 32;
    int MMA_Y      = 16;
    int MMA_X      = 8;
    int MMA_Y_HALF = MMA_Y / 2;

    // int cta_num = cta_y * cta_x / warp_y / warp_x;

    int dAvn_size = warp_y / MMA_Y_HALF;
    int dBvn_size = warp_x / MMA_X;

    std::stringstream file_str;

    file_str << "#define TILE_N_PER_CTA       " << cta_x << "\n";
    file_str << "#define TILE_M_PER_CTA       " << cta_y << "\n\n";

    file_str << "#define TILE_N_PER_WARP      " << warp_x << "\n";
    file_str << "#define TILE_M_PER_WARP      " << warp_y << "\n\n";

    file_str << "#define TILE_K_PER_CTA       " << k_size << "\n";
    file_str << "#define TILE_K_PER_STEP      " << s_size << "\n\n";

    file_str << "#define KERNEL_NAME " << kname << "\n\n";

    file_str << "#include <cuda_fp16.h>\n\n";

    file_str << "#define ENABLE_FUSE 1\n\n";
    file_str << "#define uint int\n\n";
    file_str << "#define uint32_t int\n\n";

    if (declare_times == 0) {
        file_str << "#define MAX_LUT_SIZE 128\n\n";
        file_str << "struct lut_t{ int idx[MAX_LUT_SIZE]; };\n\n";
    }

    WriteIncludeFile(file_str, "/idxn/common/const_macros.h");

    if (s_size == 8) {
        WriteIncludeFile(file_str, "/idxn/common/dmem_i1_macros.h");
        WriteIncludeFile(file_str, "/idxn/common/hmma_i1_macros.h");

        file_str << "#define LOAD_dAv1(_regA, _dAv1, _in_id, _in_off)    LOAD_dAv1_SIZE" << dAvn_size << "(_regA, _dAv1, _in_id, _in_off)\n";
        file_str << "#define LOAD_dBv1(_regB, _dBv1, _dBv1_off)          LOAD_dBv1_SIZE" << dBvn_size << "(_regB, _dBv1, _dBv1_off)\n\n";

        file_str << "#define MMA_INSTS(_C, _A, _B)                       MMA_INST_1INT_" << dAvn_size / 2 << "x" << dBvn_size << "(_C, _A, _B)\n\n";
    } else if (s_size == 16) {
        WriteIncludeFile(file_str, "/idxn/common/dmem_i2_macros.h");
        WriteIncludeFile(file_str, "/idxn/common/hmma_i2_macros.h");

        file_str << "#define LOAD_dAv2(_regA, _dAv2, _in_id, _in_off)    LOAD_dAv2_SIZE" << dAvn_size << "(_regA, _dAv2, _in_id, _in_off)\n";
        file_str << "#define LOAD_dBv2(_regB, _dBv2, _dBv2_off)          LOAD_dBv2_SIZE" << dBvn_size << "(_regB, _dBv2, _dBv2_off)\n\n";

        file_str << "#define MMA_INSTS(_C, _A, _B)                       MMA_INST_2INT_" << dAvn_size / 2 << "x" << dBvn_size << "(_C, _A, _B)\n\n";
    } else if (s_size == 32) {
        WriteIncludeFile(file_str, "/idxn/common/dmem_i4_macros.h");
        WriteIncludeFile(file_str, "/idxn/common/hmma_i4_macros.h");

        file_str << "#define LOAD_dAv4(_regA, _dAv4, _in_id, _in_off)    LOAD_dAv4_SIZE" << dAvn_size << "(_regA, _dAv4, _in_id, _in_off)\n";
        file_str << "#define LOAD_dBv4(_regB, _dBv4, _dBv4_off)          LOAD_dBv4_SIZE" << dBvn_size << "(_regB, _dBv4, _dBv4_off)\n\n";

        file_str << "#define MMA_INSTS(_C, _A, _B)                       MMA_INST_4INT_" << dAvn_size / 2 << "x" << dBvn_size << "(_C, _A, _B)\n\n";
    }

    WriteIncludeFile(file_str, "/idxn/common/output_macros.h");
    file_str << "extern \"C\" {\n\n";
    WriteIncludeFile(file_str, "/idxn/common/main_body.h");
    file_str << "}\n\n";
    WriteIncludeFile(file_str, "/idxn/common/uni_undefs.h");

    file_res = file_str.str();
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode ReplaceFusionFor2spk(std::string& file_res, fuse_info_t fuse_info)
{
    const std::set<std::string> relu_set{"Relu", "Clip", "PRelu", "LeakyRelu", "Sigmoid"};
    int fuse_index = 0;
    int fuse_size  = fuse_info.types.size();

    auto begin = file_res.find("uint concatV4_off = 0;");
    auto end   = file_res.find("#endif", begin);

    std::stringstream file_str;
    file_str << "uint concatV4_off = 0;\n";
    file_str << "if(dCv4_x_valid  && dCv4_y_valid ) {\n";

    if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        auto type = fuse_info.types[fuse_index];
        if (type == "Relu") {
            file_str << "JIT_FUSE_RELU_V4()\n";
        } else if (type == "Clip") {
            file_str << "JIT_FUSE_CLIP_V4(clip_max, clip_min)\n";
        } else if (type == "PRelu") {
            file_str << "JIT_FUSE_PRELU_V4(has_prelu, prelu)\n";
        } else if (type == "LeakyRelu") {
            file_str << "JIT_FUSE_LEAKY_V4(leaky)\n";
        } else if (type == "Sigmoid") {
            file_str << "JIT_FUSE_SIGMOID_V4()\n";
        } else {
            LOG(ERROR) << "Fuse conv with op[" << type << "] failed.";
            return ppl::common::RC_UNSUPPORTED;
        }
        fuse_index++;
    }

    if (fuse_index < fuse_size && fuse_info.types[fuse_index] == "Add") {
        file_str << "JIT_FUSE_ELT_V4(pre_data)\n";
        fuse_index++;
    }

    if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
        auto type = fuse_info.types[fuse_index];
        if (type == "Relu") {
            file_str << "JIT_FUSE_RELU_V4()\n";
        } else if (type == "Clip") {
            file_str << "JIT_FUSE_CLIP_V4(elt_clip_max, elt_clip_min)\n";
        } else if (type == "PRelu") {
            file_str << "JIT_FUSE_PRELU_V4(has_elt_prelu, elt_prelu)\n";
        } else if (type == "LeakyRelu") {
            file_str << "JIT_FUSE_LEAKY_V4(elt_leaky)\n";
        } else if (type == "Sigmoid") {
            file_str << "JIT_FUSE_SIGMOID_V4()\n";
        } else {
            LOG(ERROR) << "Fuse conv with op[" << type << "] failed.";
            return ppl::common::RC_UNSUPPORTED;
        }
        fuse_index++;
    }

    if (fuse_info.channel_offset >= 0) {
        file_str << "JIT_SET_CONCAT_OFF_V4(concatV4_off)\n";
    }

    file_str << "}\n";
    file_res.replace(begin, end - begin, file_str.str());
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode ReplaceFusionForIdxn(std::string& file_res, fuse_info_t fuse_info)
{
    const std::set<std::string> relu_set{"Relu", "Clip", "PRelu", "LeakyRelu", "Sigmoid"};
    int fuse_size = fuse_info.types.size();

    for (uint32_t i = 1; i <= 4; i *= 2) {
        int fuse_index = 0;

        auto begin = file_res.find("uint concat_v1_off0 = 0;");
        auto end   = file_res.find("#endif", begin);

        std::stringstream file_str;
        file_str << "uint concat_v1_off0 = dCv1_idy[0] * num_flt_v2;\n";
        file_str << "uint concat_v1_off1 = dCv1_idy[1] * num_flt_v2;\n";

        if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
            auto type = fuse_info.types[fuse_index];
            if (type == "Relu") {
                file_str << "JIT_FUSE_RELU_2x" << i << "_V1()\n";
            } else if (type == "Clip") {
                file_str << "JIT_FUSE_CLIP_2x" << i << "_V1(clip_max, clip_min)\n";
            } else if (type == "PRelu") {
                file_str << "JIT_FUSE_PRELU_2x" << i << "_V1(has_prelu, prelu)\n";
            } else if (type == "LeakyRelu") {
                file_str << "JIT_FUSE_LEAKY_2x" << i << "_V1(leaky)\n";
            } else if (type == "Sigmoid") {
                file_str << "JIT_FUSE_SIGMOID_2x" << i << "_V1()\n";
            } else {
                LOG(ERROR) << "Fuse conv with op[" << type << "] failed.";
                return ppl::common::RC_UNSUPPORTED;
            }
            fuse_index++;
        }

        if (fuse_index < fuse_size && fuse_info.types[fuse_index] == "Add") {
            file_str << "JIT_FUSE_ELT_2x" << i << "_V1(pre_data)\n";
            fuse_index++;
        }

        if (fuse_index < fuse_size && relu_set.find(fuse_info.types[fuse_index]) != relu_set.end()) {
            auto type = fuse_info.types[fuse_index];
            if (type == "Relu") {
                file_str << "JIT_FUSE_RELU_2x" << i << "_V1()\n";
            } else if (type == "Clip") {
                file_str << "JIT_FUSE_CLIP_2x" << i << "_V1(elt_clip_max, elt_clip_min)\n";
            } else if (type == "PRelu") {
                file_str << "JIT_FUSE_PRELU_2x" << i << "_V1(has_elt_prelu, elt_prelu)\n";
            } else if (type == "LeakyRelu") {
                file_str << "JIT_FUSE_LEAKY_2x" << i << "_V1(elt_leaky)\n";
            } else if (type == "Sigmoid") {
                file_str << "JIT_FUSE_SIGMOID_2x" << i << "_V1()\n";
            } else {
                LOG(ERROR) << "Fuse conv with op[" << type << "] failed.";
                return ppl::common::RC_UNSUPPORTED;
            }
            fuse_index++;
        }

        if (fuse_info.channel_offset >= 0) {
            file_str << "JIT_SET_CONCAT_OFF_V1(concat_v1_off0, concat_v1_off1);\n";
        }

        file_res.replace(begin, end - begin, file_str.str());
    }
    return ppl::common::RC_SUCCESS;
}