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

#ifndef __ST_PPL_KERNEL_X86_COMMON_MACROS_H_
#define __ST_PPL_KERNEL_X86_COMMON_MACROS_H_

#ifdef _MSC_VER
#define PPL_USE_X86_MSVC
#endif

#if defined(__APPLE__) && (defined(__GNUC__) || defined(__xlC__) || defined(__xlc__))
#define PPL_USE_X86_DARWIN
#endif

#if (defined(__x86_64__) && (defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)))
#ifndef __APPLE_CC__
#define PPL_USE_X86_INLINE_ASM_MACRO
#endif
#define PPL_USE_X86_INLINE_ASM
#endif

#ifdef PPL_USE_X86_MSVC
#define PPL_X86_PRAGMA(X) __pragma(X)
#else
#define PPL_X86_PRAGMA(X) _Pragma(#X)
#endif

#ifdef PPL_USE_X86_OMP
#include <omp.h>
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE(TYPE) PPL_X86_PRAGMA(omp parallel for schedule(TYPE))
#define PRAGMA_OMP_PARALLEL_FOR() PPL_X86_PRAGMA(omp parallel for)
#define PRAGMA_OMP_PARALLEL() PPL_X86_PRAGMA(omp parallel)
#define PRAGMA_OMP_FOR_NOWAIT() PPL_X86_PRAGMA(omp for nowait)
#define PRAGMA_OMP_FOR() PPL_X86_PRAGMA(omp for)
#define PRAGMA_OMP_SINGLE()   PPL_X86_PRAGMA(omp single)
#define PPL_OMP_NUM_THREADS() omp_get_num_threads()
#define PPL_OMP_MAX_THREADS() omp_get_max_threads()
#define PPL_OMP_THREAD_ID()   omp_get_thread_num()
#else
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE(TYPE)
#define PRAGMA_OMP_PARALLEL_FOR()
#define PRAGMA_OMP_PARALLEL()
#define PRAGMA_OMP_FOR_NOWAIT()
#define PRAGMA_OMP_FOR()
#define PRAGMA_OMP_SINGLE()
#define PPL_OMP_NUM_THREADS() 1
#define PPL_OMP_MAX_THREADS() 1
#define PPL_OMP_THREAD_ID()   0
#endif

#if (defined(PPL_USE_X86_OMP) && (_OPENMP >= 200805))
#define PPL_USE_X86_OMP_COLLAPSE
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(N) PPL_X86_PRAGMA(omp parallel for collapse(N))
#define PRAGMA_OMP_FOR_COLLAPSE(N) PPL_X86_PRAGMA(omp for collapse(N))
#else
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(N)
#define PRAGMA_OMP_FOR_COLLAPSE(N)
#endif

#define PPL_X86_TENSOR_MAX_DIMS() 8
#define PPL_X86_CACHELINE_BYTES() 64

#ifdef PPL_USE_X86_DARWIN
#define PPL_X86_INLINE_ASM_ALIGN() ".align 4\n"
#else
#define PPL_X86_INLINE_ASM_ALIGN() ".align 16\n"
#endif

#define PPL_STR_INNER(X) #X
#define PPL_STR(X)       PPL_STR_INNER(X)

#endif
