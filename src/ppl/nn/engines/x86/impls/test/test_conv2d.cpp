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

#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

#include <float.h>
#include <string.h>
#include <inttypes.h>

#if defined(__linux__) && defined(PPL_USE_X86_OMP)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <omp.h>
#endif

#include "ppl/kernel/x86/fp32/conv2d.h"
#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/math.h"
#include "ppl/kernel/x86/common/macros.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/nn/common/tensor_shape.h"
#include "simple_flags.h"
#include "utils/check.h"

// #define ENABLE_DEBUG_TAG
#ifdef ENABLE_DEBUG_TAG
#define DEBUG_TAG(X) fprintf(stderr, "," #X)
#else
#define DEBUG_TAG(X)
#endif

#define CASE_STRING_FMT() "g%" PRId64 "_mb%" PRId64 "_ic%" PRId64 "ih%" PRId64 "iw%" PRId64 "_oc%" PRId64 "oh%" PRId64 "ow%" PRId64 "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 "_n%s"

#define ONNX_TEST_CASE() \
"\
        ONNXTestGeneratorBase(testcase_name='%s',\n\
                            node_type='Conv',\n\
                            inputs=[ONNXTensorInfo(size=(%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "), dtype=DataType.FLOAT),\n\
                                    ONNXTensorInfo(size=(%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "), dtype=DataType.FLOAT, ttype=TensorType.INITIALIZER),\n\
                                    ONNXTensorInfo(size=(%" PRId64 "), dtype=DataType.FLOAT, ttype=TensorType.INITIALIZER)],\n\
                            outputs=[ONNXTensorInfo(dtype=DataType.FLOAT)],\n\
                            dilations=[%" PRId64 ", %" PRId64 "],\n\
                            group = %" PRId64 ",\n\
                            kernel_shape=[%" PRId64 ", %" PRId64 "],\n\
                            pads=[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "],\n\
                            strides=[%" PRId64 ", %" PRId64 "],\n\
                            use_ort_fp16 = use_ort_fp16_flag)\n\
"

Define_bool_opt("--help", Flag_help, false, "show these help information");
Define_string(cfg, "", "(required) conv config file, format:" CASE_STRING_FMT());
Define_string(algo, "", "(required) conv algorithm string");
Define_int32(loop_cfg, 1, "(1) loop config file times");
Define_int32(mb, 0, "(0) custom batch");
Define_int32(warm_up, 2, "(2) warm up iterations");
Define_int32(min_iter, 4, "(4) min benchmark iterations");
Define_float(min_second, 0.5f, "(0.5) min benchmark seconds");
Define_int32(relu, 0, "(0) fuse relu, 0,1 or 6 for relu6");
Define_bool(sum, false, "(false) fuse eltwise sum");
Define_bool(validate, false, "(false) do result validation");
Define_float(eps, 1e-6f, "(1e-6) rel error trunk for validation");
Define_bool(dynamic, false, "(false) prepare and alloc temp buffer for each run");
Define_bool(profile, false, "(false) do profile and dump profile info");
Define_bool(export_onnx_op_test, false, "(false) export cfg to ppl onnx op test format");
#ifdef PPL_USE_X86_AVX512
Define_bool(disable_avx512, false, "(false) disable avx512 for auto select algo");
#else
static bool Flag_disable_avx512 = true;
#endif
Define_bool(disable_avx_fma3, false, "(false) disable avx, fma3, avx512 for auto select algo");

/*

config file format(mkl format):
^BEG
# comment...
# comment...
case strings...\n
case strings...\n
case strings...\n
...\n
^EOF

algo string list:
n16cx_implicit_gemm_fp32_fma
n16cx_gemm_direct_fp32_fma

*/

static std::map<std::string, ppl::kernel::x86::conv2d_fp32_algo_info> algo_table =
{
    {
        "n16cx_gemm_direct_fp32_fma",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::GEMM_DIRECT,
            ppl::common::ISA_X86_FMA,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_depthwise_fp32_fma",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DEPTHWISE,
            ppl::common::ISA_X86_FMA,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_winograd_b4f3_fp32_fma",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::WINOGRAD_B4F3,
            ppl::common::ISA_X86_FMA,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_direct_fp32_fma",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DIRECT,
            ppl::common::ISA_X86_FMA,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_direct_ndarray_fp32_fma",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DIRECT,
            ppl::common::ISA_X86_FMA,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "im2col_gemm_fp32_fma",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::IM2COL_GEMM,
            ppl::common::ISA_X86_FMA,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_NDARRAY
        })
    },
#ifdef PPL_USE_X86_AVX512
    {
        "n16cx_gemm_direct_fp32_avx512",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::GEMM_DIRECT,
            ppl::common::ISA_X86_AVX512,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_depthwise_fp32_avx512",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DEPTHWISE,
            ppl::common::ISA_X86_AVX512,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_winograd_b4f3_fp32_avx512",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::WINOGRAD_B4F3,
            ppl::common::ISA_X86_AVX512,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_direct_fp32_avx512",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DIRECT,
            ppl::common::ISA_X86_AVX512,
            ppl::common::DATAFORMAT_N16CX,
            ppl::common::DATAFORMAT_N16CX
        })
    },
    {
        "n16cx_direct_ndarray_fp32_avx512",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DIRECT,
            ppl::common::ISA_X86_AVX512,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_N16CX
        })
    },
#endif
    {
        "n8cx_direct_fp32_sse",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DIRECT,
            ppl::common::ISA_X86_SSE,
            ppl::common::DATAFORMAT_N8CX,
            ppl::common::DATAFORMAT_N8CX
        })
    },
    {
        "n8cx_gemm_direct_fp32_sse",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::GEMM_DIRECT,
            ppl::common::ISA_X86_SSE,
            ppl::common::DATAFORMAT_N8CX,
            ppl::common::DATAFORMAT_N8CX
        })
    },
    {
        "n8cx_depthwise_fp32_sse",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DEPTHWISE,
            ppl::common::ISA_X86_SSE,
            ppl::common::DATAFORMAT_N8CX,
            ppl::common::DATAFORMAT_N8CX
        })
    },
    {
        "n8cx_direct_ndarray_fp32_sse",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DIRECT,
            ppl::common::ISA_X86_SSE,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_N8CX
        })
    },
    {
        "im2col_gemm_fp32_sse",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::IM2COL_GEMM,
            ppl::common::ISA_X86_SSE,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_NDARRAY
        })
    },
    {
        "depthwise_fp32_sse",
        ppl::kernel::x86::conv2d_fp32_algo_info({
            ppl::kernel::x86::conv2d_fp32_algo::DEPTHWISE,
            ppl::common::ISA_X86_SSE,
            ppl::common::DATAFORMAT_NDARRAY,
            ppl::common::DATAFORMAT_NDARRAY
        })
    },
};

int main(int argc, char **argv) {
    simple_flags::parse_args(argc, argv);
    if (Flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    ppl::kernel::x86::conv2d_fp32_algo_info algoinfo;
    ppl::common::dataformat_t src_format = ppl::common::DATATYPE_UNKNOWN;
    if (Flag_algo == "auto_ndarray") {
        src_format = ppl::common::DATAFORMAT_NDARRAY;
    }
    if (Flag_algo == "auto_n16cx") {
        src_format = ppl::common::DATAFORMAT_N16CX;
    }
    const bool auto_select_algo = src_format != ppl::common::DATATYPE_UNKNOWN;
    {
        if (!auto_select_algo) {
            auto algo_it = algo_table.find(Flag_algo);
            if (algo_it != algo_table.end()) {
                algoinfo = algo_it->second;
            } else {
                std::cerr << "algo string not found.\nsupported algo string:\n";
                for (auto it = algo_table.begin(); it != algo_table.end(); ++it) {
                    std::cerr << it->first << "\n";
                }
                std::cerr << "auto_ndarray\n";
                std::cerr << "auto_n16cx\n";
                simple_flags::print_args_info();
                return -1;
            }
        }
    }

    int32_t num_threads = 1;
#if defined(__linux__) && defined(PPL_USE_X86_OMP)
    num_threads = omp_get_max_threads();
#pragma omp parallel
    {
#define handle_error_en(en, msg) do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
        int i = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        if (int s = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
            handle_error_en(s, "pthread_setaffinity_np");
        }
#undef handle_error_en
    }
#endif

    if (Flag_relu != 0 && Flag_relu != 1 && Flag_relu != 6) {
        std::cerr << "invalid relu flag\n";
        Flag_relu = 0;
    }

    if (Flag_validate) {
        Flag_warm_up = 0;
        Flag_min_iter = 1;
        Flag_min_second = 0;
    }

    std::cerr << "==============================================================\n";
    fprintf(
        stderr,
        "num_threads=%d\ndynamic=%d\navx512=%d\nfma3=%d\nwarm_up=%d\nmin_iter=%d\nmin_second=%f\nvalidate=%d\neps=%f\nrelu=%d\nsum=%d\n",
        num_threads, Flag_dynamic, !Flag_disable_avx512, !Flag_disable_avx_fma3, Flag_warm_up, Flag_min_iter, Flag_min_second, Flag_validate, Flag_eps, Flag_relu, Flag_sum
    );

for (int64_t lcfg = 0; lcfg < Flag_loop_cfg; ++lcfg) {

    std::cerr << "==============================================================\n";
    std::cerr << "read config\n";

    std::ifstream cfgfile;
    {
        cfgfile.open(Flag_cfg, std::ios_base::in | std::ios_base::binary);
        if (!cfgfile.is_open()) {
            std::cerr << "cannot open config file\n";
            simple_flags::print_args_info();
            return -1;
        }
    }

    std::cerr << "==============================================================\n";
    std::cerr << "begin tests\n";
    std::cerr << "\%line_no,\%case_string,\%mops,\%mbs,\%min_ms,\%max_gflops,\%max_gbps,\%avg_ms,\%avg_gflops,\%avg_gbps\n";

    char line[512];
    int line_no = 0;
    int case_no = 0;
    double all_case_gflops = 0.;
    double all_case_us = 0.;
    while (cfgfile.getline(line, 512, '\n')) {
        ++line_no;

        // skip comment
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }

        char case_name[100];
        ppl::kernel::x86::conv2d_fp32_param param;
        int64_t batch;
        int64_t src_h;
        int64_t src_w;
        int64_t dst_h;
        int64_t dst_w;
        int64_t dh;
        int64_t dw;
        if (17 != sscanf(
            line,
            CASE_STRING_FMT() "\n",
            &param.group, &batch,
            &param.channels, &src_h, &src_w,
            &param.num_output, &dst_h, &dst_w,
            &param.kernel_h, &param.kernel_w,
            &param.stride_h, &param.stride_w,
            &param.pad_h, &param.pad_w,
            &dh, &dw,
            case_name
        )) {
            std::cerr << line_no << "," << line << ",invalid format\n";
            continue;
        }
        param.dilation_h = dh + 1;
        param.dilation_w = dw + 1;

        param.fuse_flag = 0;
        if (Flag_sum) {
            param.fuse_flag |= ppl::kernel::x86::conv_fuse_flag::SUM;
        }
        if (Flag_relu == 1) {
            param.fuse_flag |= ppl::kernel::x86::conv_fuse_flag::RELU;
        } else if (Flag_relu == 6) {
            param.fuse_flag |= ppl::kernel::x86::conv_fuse_flag::RELU6;
        }

        if (Flag_mb > 0) {
            batch = Flag_mb;
        }

        if (Flag_export_onnx_op_test) {
            fprintf(stderr, ONNX_TEST_CASE() "\n",
                case_name,
                batch, param.channels, src_h, src_w,
                param.num_output, param.channels,
                param.kernel_h, param.kernel_w,
                param.num_output,
                param.dilation_h, param.dilation_w,
                param.group,
                param.kernel_h, param.kernel_w,
                param.pad_h, param.pad_w, param.pad_h, param.pad_w,
                param.stride_h, param.stride_w);
            continue;
        }

        fprintf(
            stderr,
            "%d," CASE_STRING_FMT(),
            line_no,
            param.group, batch,
            param.channels, src_h, src_w,
            param.num_output, dst_h, dst_w,
            param.kernel_h, param.kernel_w,
            param.stride_h, param.stride_w,
            param.pad_h, param.pad_w,
            dh, dw,
            case_name
        );

        const int64_t ext_kernel_h = (param.kernel_h - 1) * param.dilation_h + 1;
        const int64_t ext_kernel_w = (param.kernel_w - 1) * param.dilation_w + 1;
        const int64_t assume_dst_h = ((src_h + 2 * param.pad_h - ext_kernel_h) / param.stride_h + 1);
        const int64_t assume_dst_w = ((src_w + 2 * param.pad_w - ext_kernel_w) / param.stride_w + 1);
        if (dst_h != assume_dst_h || dst_w != assume_dst_w) {
            std::cerr << "," << "dst_h(" << dst_h << ") and dst_w(" << dst_w << ") not match assume(" << assume_dst_h << ", " << assume_dst_w << ")\n";
            continue;
        }

        if (param.channels % param.group != 0 || param.num_output % param.group  != 0) {
            std::cerr << "," << "channels and num_output cannot divide by group\n";
            continue;
        }

DEBUG_TAG(A);
        ppl::common::GenericCpuAllocator allocator(PPL_X86_CACHELINE_BYTES());

        if (auto_select_algo) {
            auto isa = ppl::common::GetCpuISA();
            if (Flag_disable_avx512) {
                isa &= ~(ppl::common::ISA_X86_AVX512);
            }
            if (Flag_disable_avx_fma3) {
                isa &= ~(ppl::common::ISA_X86_AVX512);
                isa &= ~(ppl::common::ISA_X86_FMA);
                isa &= ~(ppl::common::ISA_X86_AVX);
            }
            algoinfo = ppl::kernel::x86::conv2d_algo_selector::select_algo(src_format, param, isa);
            if (algoinfo.algo_type == ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
                std::cerr << "," << "unsupported case\n";
                continue;
            }
        }

        auto conv_mgr = ppl::kernel::x86::conv2d_algo_selector::gen_algo(param, algoinfo, &allocator);

        if (!conv_mgr->is_supported()) {
            delete conv_mgr;
            std::cerr << "," << "unsupported case\n";
            continue;
        }
DEBUG_TAG(B);

        const int32_t wei_mod = 7;
        const int32_t src_mod = 5;
        const int32_t wei_shift = -3;
        const int32_t src_shift = -2;
        const float wei_scale = Flag_validate ? 1.0 : 0.1;
        const float src_scale = Flag_validate ? 1.0 : 0.1;

        const int64_t ic = param.channels / param.group;
        const int64_t oc = param.num_output / param.group;
        const float gops = param.group * batch * ic * oc * param.kernel_h * param.kernel_w * dst_h * dst_w * 2.0f / 1e9f;

DEBUG_TAG(C);
        ppl::nn::TensorShape src_shape;
        src_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        src_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        src_shape.Reshape({batch, param.channels, src_h, src_w});

        ppl::nn::TensorShape src_trans_shape = src_shape;
        if (algoinfo.input_format != ppl::common::DATAFORMAT_NDARRAY) {
            src_trans_shape.SetDataFormat(algoinfo.input_format);
        }

        ppl::nn::TensorShape dst_shape;
        dst_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        dst_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        dst_shape.Reshape({batch, param.num_output, dst_h, dst_w});

        ppl::nn::TensorShape dst_trans_shape = dst_shape;
        if (algoinfo.output_format != ppl::common::DATAFORMAT_NDARRAY) {
            dst_trans_shape.SetDataFormat(algoinfo.output_format);
        }

        ppl::nn::TensorShape filter_shape;
        filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        filter_shape.Reshape({param.num_output, param.channels / param.group, param.kernel_h, param.kernel_w});

        ppl::nn::TensorShape bias_shape;
        bias_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        bias_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        bias_shape.Reshape({param.num_output});

        const float mbs = ((float)src_shape.GetBytesExcludingPadding() +
                          dst_shape.GetBytesExcludingPadding() +
                          (Flag_sum ? dst_shape.GetBytesExcludingPadding() : 0) +
                          filter_shape.GetBytesExcludingPadding() +
                          bias_shape.GetBytesExcludingPadding()) / 1024 / 1024;

DEBUG_TAG(D);
        float *src = nullptr;
        float *dst = nullptr;
        float *sum_src = nullptr;
        float *dst_ref = nullptr;
        float *src_trans = nullptr;
        float *dst_trans = nullptr;
        float *sum_src_trans = nullptr;
        float *filter = nullptr;
        float *bias = nullptr;
        void *temp_buffer = nullptr;
        src = (float*)allocator.Alloc(src_shape.GetBytesIncludingPadding());
        filter = (float*)allocator.Alloc(filter_shape.GetBytesIncludingPadding());
        bias = (float*)allocator.Alloc(bias_shape.GetBytesIncludingPadding());
        if (!src || !filter || !bias) {
            std::cerr << "," << "input tensors out of memory\n";
            return -1;
        }
DEBUG_TAG(E);
        for (uint64_t i = 0; i < filter_shape.GetElementsIncludingPadding(); ++i) {
            filter[i] = (rand() % wei_mod + wei_shift) * wei_scale;
        }
        for (uint64_t i = 0; i < bias_shape.GetElementsIncludingPadding(); ++i) {
            bias[i] = (rand() % wei_mod + wei_shift) * wei_scale * 10.0f;
        }
        for (uint64_t i = 0; i < src_shape.GetElementsIncludingPadding(); ++i) {
            src[i] = (rand() % src_mod + src_shift) * src_scale;
        }

DEBUG_TAG(F);
        if (algoinfo.input_format != ppl::common::DATAFORMAT_NDARRAY) {
            src_trans = (float*)allocator.Alloc(src_trans_shape.GetBytesIncludingPadding());
            if (!src_trans) {
                std::cerr << "," << "src_trans out of memory\n";
                return -1;
            }
            ppl::common::RetCode ret_code = ppl::common::RC_INVALID_VALUE;
            if (algoinfo.input_format == ppl::common::DATAFORMAT_N16CX) {
                ret_code = ppl::kernel::x86::reorder_ndarray_n16cx_fp32_avx(&src_shape, src, src_trans);
            }
            if (algoinfo.input_format == ppl::common::DATAFORMAT_N8CX) {
                ret_code = ppl::kernel::x86::reorder_ndarray_n8cx_fp32(&src_shape, src, src_trans);
            }
            if (ppl::common::RC_SUCCESS != ret_code) {
                std::cerr << "," << " reorder src_trans failed\n";
                return -1;
            }
        }
DEBUG_TAG(G);
        if (algoinfo.output_format == ppl::common::DATAFORMAT_NDARRAY || Flag_validate) {
            dst = (float*)allocator.Alloc(dst_shape.GetBytesIncludingPadding());
            if (!dst) {
                std::cerr << "," << "dst out of memory\n";
                return -1;
            }
            memset(dst, 0, dst_shape.GetBytesIncludingPadding());
        }
DEBUG_TAG(H);
        if (Flag_validate) {
            dst_ref = (float*)allocator.Alloc(dst_shape.GetBytesIncludingPadding());
            if (!dst_ref) {
                std::cerr << "," << "dst_ref out of memory\n";
                return -1;
            }
            memset(dst_ref, 0, dst_shape.GetBytesIncludingPadding());
        }
DEBUG_TAG(I);
        if (algoinfo.output_format != ppl::common::DATAFORMAT_NDARRAY) {
            dst_trans = (float*)allocator.Alloc(dst_trans_shape.GetBytesIncludingPadding());
            if (!dst_trans) {
                std::cerr << "," << "dst_trans out of memory\n";
                return -1;
            }
            memset(dst_trans, 0, dst_trans_shape.GetBytesIncludingPadding());
        }
        if (Flag_sum) {
            sum_src = (float*)allocator.Alloc(dst_shape.GetBytesIncludingPadding());
            if (!sum_src) {
                std::cerr << "," << "sum_src out of memory\n";
                return -1;
            }
            for (uint64_t i = 0; i < dst_shape.GetElementsIncludingPadding(); ++i) {
                sum_src[i] = (rand() % src_mod + src_shift) * src_scale;
            }
            if (algoinfo.output_format != ppl::common::DATAFORMAT_NDARRAY) {
                sum_src_trans = (float*)allocator.Alloc(dst_trans_shape.GetBytesIncludingPadding());
                if (!sum_src_trans) {
                    std::cerr << "," << "sum_src_trans out of memory\n";
                    return -1;
                }
                ppl::common::RetCode ret_code = ppl::common::RC_INVALID_VALUE;
                if (algoinfo.output_format == ppl::common::DATAFORMAT_N16CX) {
                    ret_code = ppl::kernel::x86::reorder_ndarray_n16cx_fp32_avx(&dst_shape, sum_src, sum_src_trans);
                }
                if (algoinfo.output_format == ppl::common::DATAFORMAT_N8CX) {
                    ret_code = ppl::kernel::x86::reorder_ndarray_n8cx_fp32(&dst_shape, sum_src, sum_src_trans);
                }
                if (ppl::common::RC_SUCCESS != ret_code) {
                    std::cerr << "," << "reorder sum_src_trans failed\n";
                    return -1;
                }
            }
        }

DEBUG_TAG(J);
        if (ppl::common::RC_SUCCESS != conv_mgr->gen_cvt_weights(filter, bias)) {
            std::cerr << "," << "gen_cvt_weights failed\n";
            return -1;
        }

DEBUG_TAG(K);
        auto conv_exe = conv_mgr->gen_executor();
        conv_exe->set_src_shape(&src_shape);
        conv_exe->set_dst_shape(&dst_shape);
        conv_exe->set_sum_src_shape(&dst_shape);

        if (ppl::common::RC_SUCCESS != conv_exe->prepare()) {
            std::cerr << "," << "prepare failed\n";
            return -1;
        }
DEBUG_TAG(L);
        if (!Flag_dynamic) {
            const uint64_t temp_buffer_size = conv_exe->cal_temp_buffer_size();
            temp_buffer = allocator.Alloc(temp_buffer_size);
            if (!temp_buffer) {
                std::cerr << "," << "temp_buffer out of memory\n";
                return -1;
            }
            memset(temp_buffer, 0, temp_buffer_size);
            conv_exe->set_temp_buffer(temp_buffer);
        }
DEBUG_TAG(M);
        if (algoinfo.input_format == ppl::common::DATAFORMAT_NDARRAY) {
            conv_exe->set_src(src);
        } else {
            conv_exe->set_src(src_trans);
        }
        if (algoinfo.output_format == ppl::common::DATAFORMAT_NDARRAY) {
            conv_exe->set_sum_src(sum_src);
            conv_exe->set_dst(dst);
        } else {
            conv_exe->set_sum_src(sum_src_trans);
            conv_exe->set_dst(dst_trans);
        }

DEBUG_TAG(N);
        const bool with_profiler = conv_exe->init_profiler();
        for (int32_t i = 0; i < Flag_warm_up; ++i) {
            if (Flag_dynamic) {
                conv_exe->prepare();
                conv_exe->set_temp_buffer(allocator.Alloc(conv_exe->cal_temp_buffer_size()));
            }
            if (ppl::common::RC_SUCCESS != conv_exe->execute()) {
                std::cerr << "," << "execute failed\n";
                return -1;
            }
            if (Flag_dynamic) {
                allocator.Free(conv_exe->temp_buffer());
            }
        }

        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;
        double tot_exe_us = 0.;
        double min_exe_us = DBL_MAX;
        int64_t tot_exe_iter = 0;

        conv_exe->clear_profiler();

        for (; tot_exe_iter < Flag_min_iter || tot_exe_us < Flag_min_second * 1e6; ++tot_exe_iter) {
            start = std::chrono::high_resolution_clock::now();
            if (Flag_dynamic) {
                conv_exe->prepare();
                conv_exe->set_temp_buffer(allocator.Alloc(conv_exe->cal_temp_buffer_size()));
            }
            conv_exe->execute();
            if (Flag_dynamic) {
                allocator.Free(conv_exe->temp_buffer());
            }
            end = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e3;
            tot_exe_us += dur;
            if (dur < min_exe_us) {
                min_exe_us = dur;
            }
        }

        std::string profile_result = Flag_profile ? conv_exe->export_profiler() : "";

        double avg_exe_us = tot_exe_us / tot_exe_iter;
        double max_gflops = gops / (min_exe_us / 1e6);
        double avg_gflops = gops / (avg_exe_us / 1e6);
        double max_gbps = mbs / 1024 / (min_exe_us / 1e6);
        double avg_mbs = mbs / 1024 / (avg_exe_us / 1e6);
        fprintf(stderr, ",%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.2f,%.2f", gops * 1000, mbs, min_exe_us / 1e3, max_gflops, max_gbps, avg_exe_us / 1e3, avg_gflops, avg_mbs);

        ++case_no;
        all_case_gflops += avg_gflops;
        all_case_us += avg_exe_us;

DEBUG_TAG(O);
        if (Flag_validate) {
            if (ppl::common::RC_SUCCESS != ppl::kernel::x86::conv2d_ref_fp32(
                    &src_shape,
                    &dst_shape,
                    &dst_shape,
                    src,
                    sum_src,
                    filter,
                    bias,
                    param,
                    dst_ref)) {
                std::cerr << "," << "conv2d_ref_fp32 failed\n";
                return -1;
            }
            if (algoinfo.output_format != ppl::common::DATAFORMAT_NDARRAY) {
                ppl::common::RetCode ret_code = ppl::common::RC_INVALID_VALUE;
                if (algoinfo.output_format == ppl::common::DATAFORMAT_N16CX) {
                    ret_code = ppl::kernel::x86::reorder_n16cx_ndarray_fp32_avx(&dst_trans_shape, dst_trans, dst);
                }
                if (algoinfo.output_format == ppl::common::DATAFORMAT_N8CX) {
                    ret_code = ppl::kernel::x86::reorder_n8cx_ndarray_fp32(&dst_trans_shape, dst_trans, dst);
                }
                if (ppl::common::RC_SUCCESS != ret_code) {
                    std::cerr << "," << "reorder dst_trans failed\n";
                    return -1;
                }
            }
            std::cerr << ",";
            check_array_error(dst, dst_ref, dst_shape.GetElementsIncludingPadding(), Flag_eps);
        }

        if (Flag_profile && with_profiler) {
            std::cerr << "\n";
            std::cerr << profile_result;
        }

DEBUG_TAG(Y);
        conv_mgr->release_cvt_weights();
        if (conv_mgr) delete conv_mgr;
        if (conv_exe) delete conv_exe;
        if (src) allocator.Free(src);
        if (filter) allocator.Free(filter);
        if (bias) allocator.Free(bias);
        if (dst) allocator.Free(dst);
        if (sum_src) allocator.Free(sum_src);
        if (dst_ref) allocator.Free(dst_ref);
        if (src_trans) allocator.Free(src_trans);
        if (dst_trans) allocator.Free(dst_trans);
        if (sum_src_trans) allocator.Free(sum_src_trans);
        if (!Flag_dynamic) {
            if (temp_buffer) allocator.Free(temp_buffer);
        }
DEBUG_TAG(Z);
        std::cerr << "\n";
    }
    std::cerr << "tot time(ms): " << all_case_us / 1e3 << "\t" << "avg gflops: " << all_case_gflops / case_no << "\n";
    cfgfile.close();
}

}
