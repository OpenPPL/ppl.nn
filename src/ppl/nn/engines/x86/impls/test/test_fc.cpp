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

#include <inttypes.h>
#include <float.h>
#include <string.h>

#if defined(__linux__) && defined(PPL_USE_X86_OMP)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <omp.h>
#endif

#include "ppl/kernel/x86/fp32/fc.h"
#include "ppl/kernel/x86/fp32/gemm.h"
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

#define CASE_STRING_FMT() "m%" PRId64 "n%" PRId64 "k%" PRId64 "_n%s"

Define_bool_opt("--help", Flag_help, false, "show these help information");
Define_string(cfg, "", "(required) fc config file, format:" CASE_STRING_FMT());
Define_int32(mb, 0, "(0) custom batch");
Define_int32(warm_up, 10, "(10) warm up iterations");
Define_int32(min_iter, 20, "(20) min benchmark iterations");
Define_float(min_second, 1.0f, "(1.0) min benchmark seconds");
Define_bool(validate, false, "(false) do result validation");
Define_float(eps, 1e-6f, "(1e-6) rel error trunk for validation");

int main(int argc, char **argv) {
    simple_flags::parse_args(argc, argv);
    if (Flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

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

    if (Flag_validate) {
        Flag_warm_up = 0;
        Flag_min_iter = 1;
        Flag_min_second = 0;
    }

    std::cerr << "==============================================================\n";
    fprintf(
        stderr,
        "num_threads=%d\nwarm_up=%d\nmin_iter=%d\nmin_second=%f\nvalidate=%d\neps=%f\n\n",
        num_threads, Flag_warm_up, Flag_min_iter, Flag_min_second, Flag_validate, Flag_eps
    );
    std::cerr << "==============================================================\n";
    std::cerr << "begin tests\n";
    std::cerr << "line_no,case_string,min_ms,max_gflops,max_gbps,avg_ms,avg_gflops,avg_gbps\n";

    char line[512];
    int line_no = 0;
    int case_no = 0;
    double all_case_gflops = 0.;
    double all_case_gbps = 0.;
    double all_case_us = 0.;
    while (cfgfile.getline(line, 512, '\n')) {
        ++line_no;

        // skip comment
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }

        char case_name[100];
        ppl::kernel::x86::fc_fp32_param param;
        int64_t M, N, K;
        if (4 != sscanf(
            line,
            CASE_STRING_FMT() "\n",
            &M, &N, &K, case_name
        )) {
            std::cerr << line_no << "," << line << ",invalid format\n";
            continue;
        }
        param.channels = K;
        param.num_output = N;
        if (Flag_mb) {
            M = Flag_mb;
        }

        fprintf(
            stderr,
            "%d," CASE_STRING_FMT(),
            line_no, M, N, K, case_name
        );

DEBUG_TAG(A);
        ppl::common::GenericCpuAllocator allocator(PPL_X86_CACHELINE_BYTES());
        ppl::kernel::x86::fc_fp32_algo_info algoinfo;

        algoinfo = ppl::kernel::x86::fc_algo_selector::select_algo(ppl::common::DATAFORMAT_NDARRAY, param, ppl::common::GetCpuISA());
        if (algoinfo.algo_type == ppl::kernel::x86::fc_fp32_algo::UNKNOWN) {
            std::cerr << "," << "unsupported case\n";
            continue;
        }
        auto fc_mgr = ppl::kernel::x86::fc_algo_selector::gen_algo(param, algoinfo, &allocator);

DEBUG_TAG(B);

        const int32_t wei_mod = 7;
        const int32_t src_mod = 5;
        const int32_t wei_shift = -3;
        const int32_t src_shift = -2;
        const float wei_scale = Flag_validate ? 1.0 : 0.1;
        const float src_scale = Flag_validate ? 1.0 : 0.1;

        const float gops = static_cast<float>(M) * N * K * 2.0f / 1e9f;
        const float gbs = static_cast<float>(M * K + N * K + M * N) * sizeof(float) / 1024 / 1024 / 1024;

DEBUG_TAG(C);
        ppl::nn::TensorShape src_shape;
        src_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        src_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        src_shape.Reshape({M, K});

        ppl::nn::TensorShape dst_shape;
        dst_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        dst_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        dst_shape.Reshape({M, N});

        ppl::nn::TensorShape filter_shape;
        filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        filter_shape.Reshape({N, K});

        ppl::nn::TensorShape bias_shape;
        bias_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        bias_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        bias_shape.Reshape({N});

DEBUG_TAG(D);
        float *src = nullptr;
        float *dst = nullptr;
        float *dst_ref = nullptr;
        float *filter = nullptr;
        float *bias = nullptr;
        void *temp_buffer = nullptr;
        src = (float*)allocator.Alloc(src_shape.GetBytesIncludingPadding());
        filter = (float*)allocator.Alloc(filter_shape.GetBytesIncludingPadding());
        bias = (float*)allocator.Alloc(bias_shape.GetBytesIncludingPadding());
        dst = (float*)allocator.Alloc(dst_shape.GetBytesIncludingPadding());

        if (!src || !filter || !bias || !dst) {
            std::cerr << "," << "input & output tensors out of memory\n";
            return -1;
        }
DEBUG_TAG(E);
        for (uint64_t i = 0; i < src_shape.GetElementsIncludingPadding(); ++i) {
            src[i] = (rand() % src_mod + src_shift) * src_scale;
        }
        for (uint64_t i = 0; i < filter_shape.GetElementsIncludingPadding(); ++i) {
            filter[i] = (rand() % wei_mod + wei_shift) * wei_scale;
        }
        for (uint64_t i = 0; i < bias_shape.GetElementsIncludingPadding(); ++i) {
            bias[i] = (rand() % wei_mod + wei_shift) * wei_scale;
        }
        memset(dst, 0, dst_shape.GetBytesIncludingPadding());

DEBUG_TAG(H);
        if (Flag_validate) {
            dst_ref = (float*)allocator.Alloc(dst_shape.GetBytesIncludingPadding());
            if (!dst_ref) {
                std::cerr << "," << "dst_ref out of memory\n";
                return -1;
            }
            memset(dst_ref, 0, dst_shape.GetBytesIncludingPadding());
        }

DEBUG_TAG(J);
        if (ppl::common::RC_SUCCESS != fc_mgr->gen_cvt_weights(filter, bias)) {
            std::cerr << "," << "gen_cvt_weights failed\n";
            return -1;
        }

DEBUG_TAG(K);
        auto fc_exe = fc_mgr->gen_executor();
        fc_exe->set_src_shape(&src_shape);
        fc_exe->set_dst_shape(&dst_shape);

        if (ppl::common::RC_SUCCESS != fc_exe->prepare()) {
            std::cerr << "," << "prepare failed\n";
            return -1;
        }
DEBUG_TAG(L);
        const uint64_t temp_buffer_size = fc_exe->cal_temp_buffer_size();
        temp_buffer = allocator.Alloc(temp_buffer_size);
        if (!temp_buffer) {
            std::cerr << "," << "temp_buffer out of memory\n";
            return -1;
        }
        memset(temp_buffer, 0, temp_buffer_size);
        fc_exe->set_temp_buffer(temp_buffer);
        fc_exe->set_src(src);
        fc_exe->set_dst(dst);

DEBUG_TAG(N);
        const bool with_profiler = fc_exe->init_profiler();
        for (int32_t i = 0; i < Flag_warm_up; ++i) {
            if (ppl::common::RC_SUCCESS != fc_exe->execute()) {
                std::cerr << "," << "execute failed\n";
                return -1;
            }
        }

        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;
        double tot_exe_us = 0.;
        double min_exe_us = DBL_MAX;
        int64_t tot_exe_iter = 0;

        fc_exe->clear_profiler();

        for (; tot_exe_iter < Flag_min_iter || tot_exe_us < Flag_min_second * 1e6; ++tot_exe_iter) {
            start = std::chrono::high_resolution_clock::now();
            if (ppl::common::RC_SUCCESS != fc_exe->execute()) {
                std::cerr << "," << "execute failed\n";
                return -1;
            }
            end = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e3;
            tot_exe_us += dur;
            if (dur < min_exe_us) {
                min_exe_us = dur;
            }
        }

        std::string profile_result = fc_exe->export_profiler();

        double avg_exe_us = tot_exe_us / tot_exe_iter;
        double max_gflops = gops / (min_exe_us / 1e6);
        double avg_gflops = gops / (avg_exe_us / 1e6);
        double max_gbps = gbs / (min_exe_us / 1e6);
        double avg_gbps = gbs / (avg_exe_us / 1e6);
        fprintf(stderr, ",%.3f,%.2f,%.2f,%.3f,%.2f,%.2f",
            min_exe_us / 1e3, max_gflops, max_gbps,
            avg_exe_us / 1e3, avg_gflops, avg_gbps);

        ++case_no;
        all_case_gflops += avg_gflops;
        all_case_gbps += avg_gbps;
        all_case_us += avg_exe_us;

DEBUG_TAG(O);
        if (Flag_validate) {
            if (ppl::common::RC_SUCCESS != ppl::kernel::x86::gemm_ref_fp32(
                    src, filter, bias, nullptr,
                    ppl::kernel::x86::gemm_m_type::NOTRANS,
                    ppl::kernel::x86::gemm_m_type::TRANS,
                    ppl::kernel::x86::gemm_v_type::ROW_VEC,
                    ppl::kernel::x86::gemm_m_type::EMPTY,
                    M, N, K, K, K, N, 0,
                    1.0f, 1.0f,
                    ppl::kernel::x86::gemm_post::NONE,
                    dst_ref)) {
                std::cerr << "," << "gemm_ref_fp32 failed\n";
                return -1;
            }
            std::cerr << ",";
            check_array_error(dst, dst_ref, dst_shape.GetElementsIncludingPadding(), Flag_eps);
        }

        if (with_profiler) {
            std::cerr << "\n";
            std::cerr << profile_result;
        }

DEBUG_TAG(Y);
        fc_mgr->release_cvt_weights();
        if (fc_mgr) delete fc_mgr;
        if (fc_exe) delete fc_exe;
        if (src) allocator.Free(src);
        if (filter) allocator.Free(filter);
        if (bias) allocator.Free(bias);
        if (dst) allocator.Free(dst);
        if (dst_ref) allocator.Free(dst_ref);
        if (temp_buffer) allocator.Free(temp_buffer);
DEBUG_TAG(Z);
        std::cerr << "\n";
    }
    std::cerr
        << "tot time(ms): "<< all_case_us / 1e3 << "\t"
        << "avg gflops: " << all_case_gflops / case_no  << "\t"
        << "avg gbps: " << all_case_gbps / case_no << "\n";
    cfgfile.close();
}
