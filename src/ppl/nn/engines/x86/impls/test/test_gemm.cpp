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
#include <fstream>
#include <float.h>
#include <string.h>
#include <chrono>
#include <memory>
#include <inttypes.h>

#if defined(__linux__) && defined(PPL_USE_X86_OMP)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <omp.h>
#endif

#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/kernel/x86/common/simd_tools.h"
#include "ppl/kernel/x86/common/internal_include.h"
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
Define_int32(warm_up, 10, "(10) warm up iterations");
Define_int32(min_iter, 20, "(20) min benchmark iterations");
Define_float(min_second, 1.0f, "(1.0) min benchmark seconds");
Define_bool(validate, false, "(false) do result validation");
Define_float(eps, 1e-5f, "(1e-5) rel error trunk for validation");

Define_float(alpha, 1.0f, "(1.0) gemm alpha");
Define_float(beta, 0.0f, "(0.0) gemm beta");
Define_int32(relu, 0, "(0) fuse relu, 0 for none, 1 for relu, 6 for relu6");
Define_int32(type_a, 0, "(0) 0 for no_trans, 1 for trans");
Define_int32(type_b, 0, "(0) 0 for no_trans, 1 for trans");
Define_int32(type_v, 0, "(0) 0 for empty, 1 for scalar, 2 for col vector, 3 for row vector");
Define_int32(type_h, 0, "(0) 0 for empty, 1 for no_trans");
Define_int32(m, 0, "(0) override M");
Define_int32(n, 0, "(0) override N");
Define_int32(k, 0, "(0) override K");

int main(int argc, char **argv) {
    simple_flags::parse_args(argc, argv);
    if (Flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    ppl::kernel::x86::set_denormals_zero(1);

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
    fprintf(
        stderr,
        "alpha=%f\nbeta=%f\nrelu=%d\ntype_a=%d\ntype_b=%d\ntype_v=%d\ntype_h=%d\nM=%d\nN=%d\nK=%d\n\n",
        Flag_alpha, Flag_beta, Flag_relu, Flag_type_a, Flag_type_b, Flag_type_v, Flag_type_h, Flag_m, Flag_n, Flag_k
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

    ppl::kernel::x86::gemm_m_type_t typeA;
    ppl::kernel::x86::gemm_m_type_t typeB;
    ppl::kernel::x86::gemm_v_type_t typeV;
    ppl::kernel::x86::gemm_m_type_t typeH;
    ppl::kernel::x86::gemm_post_t post_flag;

    switch (Flag_type_a) {
        case 1: typeA = ppl::kernel::x86::gemm_m_type::TRANS; break;
        default: typeA = ppl::kernel::x86::gemm_m_type::NOTRANS; break;
    }
    const bool is_trans_a = Flag_type_a == 1;

    switch (Flag_type_b) {
        case 1: typeB = ppl::kernel::x86::gemm_m_type::TRANS; break;
        default: typeB = ppl::kernel::x86::gemm_m_type::NOTRANS; break;
    }
    const bool is_trans_b = Flag_type_b == 1;

    switch (Flag_type_v) {
        case 1: typeV = ppl::kernel::x86::gemm_v_type::SCALAR; break;
        case 2: typeV = ppl::kernel::x86::gemm_v_type::COL_VEC; break;
        case 3: typeV = ppl::kernel::x86::gemm_v_type::ROW_VEC; break;
        default: typeV = ppl::kernel::x86::gemm_v_type::EMPTY; break;
    }

    switch (Flag_type_h) {
        case 1: typeH = ppl::kernel::x86::gemm_m_type::NOTRANS; break;
        default: typeH = ppl::kernel::x86::gemm_m_type::EMPTY; break;
    }

    switch (Flag_relu) {
        case 1: post_flag = ppl::kernel::x86::gemm_post::RELU; break;
        case 6: post_flag = ppl::kernel::x86::gemm_post::RELU6; break;
        default: post_flag = ppl::kernel::x86::gemm_post::NONE; break;
    }

    const int32_t data_mod = 7;
    const int32_t data_shift = -3;
    const float data_scale = Flag_validate ? 1.0 : 0.1;

    while (cfgfile.getline(line, 512, '\n')) {
        ++line_no;

        // skip comment
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }

        char case_name[100];
        int64_t M, N, K;

        if (4 != sscanf(
            line,
            CASE_STRING_FMT() "\n",
            &M, &N, &K, case_name
        )) {
            std::cerr << line_no << "," << line << ",invalid format\n";
            continue;
        }

        if (Flag_m > 0) M = Flag_m;
        if (Flag_n > 0) N = Flag_n;
        if (Flag_k > 0) K = Flag_k;

        fprintf(
            stderr,
            "%d," CASE_STRING_FMT(),
            line_no, M, N, K, case_name
        );

DEBUG_TAG(A);
        const int64_t lda = is_trans_a ? M : K;
        const int64_t ldb = is_trans_b ? K : N;
        const int64_t ldh = N;
        const int64_t ldc = N;
        const int64_t A_num_elements = M * K;
        const int64_t B_num_elements = K * N;
        const int64_t C_num_elements = M * N;
        const int64_t H_num_elements = Flag_type_h ? C_num_elements : 0;
        const int64_t A_num_bytes = A_num_elements * sizeof(float);
        const int64_t B_num_bytes = B_num_elements * sizeof(float);
        const int64_t C_num_bytes = C_num_elements * sizeof(float);
        const int64_t H_num_bytes = Flag_type_h ? C_num_bytes : 0;

        int64_t V_num_elements;
        switch (Flag_type_v) {
            case 1: V_num_elements = 1; break;
            case 2: V_num_elements = M; break;
            case 3: V_num_elements = N; break;
            default: V_num_elements = 0; break;
        }
        const int64_t V_num_bytes = V_num_elements * sizeof(float);

        const double gops = (double)M * N * K * 2 / 1e9;
        const double gbs = (double)(A_num_bytes + B_num_bytes + C_num_bytes + V_num_bytes + H_num_bytes) / 1e9;
DEBUG_TAG(B);
        ppl::common::GenericCpuAllocator allocator(PPL_X86_CACHELINE_BYTES());

        float* A = (float*)allocator.Alloc(A_num_bytes);
        float* B = (float*)allocator.Alloc(B_num_bytes);
        float* C = (float*)allocator.Alloc(C_num_bytes);
        if (!A || !B || !C) {
            std::cerr << ", A or B or C out of memory\n";
            return -1;
        }
        memset(C, 0, C_num_bytes);

        float* V = nullptr;
        if (V_num_bytes > 0) {
            V = (float*)allocator.Alloc(V_num_bytes);
            if (!V) {
                std::cerr << ", V out of memory\n";
                return -1;
            }
        }

        float* H = nullptr;
        if (H_num_bytes > 0) {
            H = (float*)allocator.Alloc(H_num_bytes);
            if (!H) {
                std::cerr << ", H out of memory\n";
                return -1;
            }
        }

DEBUG_TAG(C);
        for (int64_t i = 0; i < A_num_elements; i++) {
            A[i] = (rand() % data_mod + data_shift) * data_scale;
        }
        for (int64_t i = 0; i < B_num_elements; i++) {
            B[i] = (rand() % data_mod + data_shift) * data_scale;
        }
        for (int64_t i = 0; i < V_num_elements; i++) {
            V[i] = (rand() % data_mod + data_shift) * data_scale;
        }
        for (int64_t i = 0; i < H_num_elements; i++) {
            H[i] = (rand() % data_mod + data_shift) * data_scale;
        }

        float* C_ref = nullptr;
        if (Flag_validate) {
            C_ref = (float*)allocator.Alloc(C_num_bytes);
            if (!C_ref) {
                std::cerr << "," << "C_ref out of memory\n";
                return -1;
            }
            memset(C_ref, 0, C_num_bytes);
        }
DEBUG_TAG(D);
        ppl::common::RetCode ret =
            ppl::kernel::x86::gemm_fp32_fma(
                A, B, V, H,
                typeA, typeB, typeV, typeH,
                M, N, K,
                lda, ldb, ldc, ldh,
                Flag_alpha, Flag_beta,
                post_flag, C);
        if (ret != ppl::common::RC_SUCCESS) {
            fprintf(stderr, "execute failed!\n");
            return -1;
        }

DEBUG_TAG(G);
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;
        double tot_exe_us = 0.;
        double min_exe_us = DBL_MAX;
        int64_t tot_exe_iter = 0;

        for (; tot_exe_iter < Flag_min_iter || tot_exe_us < Flag_min_second * 1e6; ++tot_exe_iter) {
            start = std::chrono::high_resolution_clock::now();
            ppl::kernel::x86::gemm_fp32_fma(
                A, B, V, H,
                typeA, typeB, typeV, typeH,
                M, N, K,
                lda, ldb, ldc, ldh,
                Flag_alpha, Flag_beta,
                post_flag, C);
            end = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e3;
            tot_exe_us += dur;
            if (dur < min_exe_us) {
                min_exe_us = dur;
            }
        }

        double avg_exe_us = tot_exe_us / tot_exe_iter;
        double max_gflops = gops / (min_exe_us / 1e6);
        double avg_gflops = gops / (avg_exe_us / 1e6);
        double max_gbps = gbs / (min_exe_us / 1e6);
        double avg_gbps = gbs / (avg_exe_us / 1e6);

        fprintf(stderr, ",%.3f,%.2f,%.2f,%.3f,%.2f,%.2f",
            min_exe_us / 1e3, max_gflops, max_gbps,
            avg_exe_us / 1e3, avg_gflops, avg_gbps);

        if (Flag_validate) {
            if (ppl::common::RC_SUCCESS != ppl::kernel::x86::gemm_ref_fp32(
                A, B, V, H,
                typeA, typeB, typeV, typeH,
                M, N, K,
                lda, ldb, ldc, ldh,
                Flag_alpha, Flag_beta,
                post_flag, C_ref)) {
                std::cerr << "," << "gemm_ref_fp32 failed\n";
                return -1;
            }
            std::cerr << ",";
            check_array_error(C, C_ref, C_num_elements, Flag_eps);
        }

        ++case_no;
        all_case_gflops += avg_gflops;
        all_case_gbps += avg_gbps;
        all_case_us += avg_exe_us;
DEBUG_TAG(H);
        allocator.Free(A);
        allocator.Free(B);
        allocator.Free(C);
        if (V) {
            allocator.Free(V);
        }
        if (H) {
            allocator.Free(H);
        }
        if (C_ref) {
            allocator.Free(C_ref);
        }
        std::cerr << "\n";
    }
    std::cerr
        << "tot time(ms): "<< all_case_us / 1e3 << "\t"
        << "avg gflops: " << all_case_gflops / case_no  << "\t"
        << "avg gbps: " << all_case_gbps / case_no << "\n";
    cfgfile.close();

    return 0;
}
