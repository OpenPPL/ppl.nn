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

// pred ca type
#define PRED_CP_ASYNC_CA(_pred, _sm_v4, _gm_v4, _cp_size) \
        asm volatile( \
                "{\n" \
                " .reg .pred p;\n" \
                " setp.ne.b32 p, %0, 0;\n" \
                " @p cp.async.ca.shared.global [%1], [%2], %3;\n" \
                "}\n" :: \
                "r"((int)_pred), "r"(_sm_v4), "l"(_gm_v4), "n"(_cp_size));

#define PRED_CP_ASYNC_CA_L2_PREFETCH(_pred, _sm_v4, _gm_v4, _cp_size) \
        asm volatile( \
                "{\n" \
                " .reg .pred p;\n" \
                " setp.ne.b32 p, %0, 0;\n" \
                " @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n" \
                "}\n" :: \
                "r"((int)_pred), "r"(_sm_v4), "l"(_gm_v4), "n"(_cp_size));

// pred cg type
#define PRED_CP_ASYNC_CG(_pred, _sm_v4, _gm_v4) \
        asm volatile( \
                "{\n" \
                " .reg .pred p;\n" \
                " setp.ne.b32 p, %0, 0;\n" \
                " @p cp.async.cg.shared.global [%1], [%2], %3;\n" \
                "}\n" :: \
                "r"((int)_pred), "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_));

#define PRED_CP_ASYNC_CG_L2_PREFETCH(_pred, _sm_v4, _gm_v4) \
        asm volatile( \
                "{\n" \
                " .reg .pred p;\n" \
                " setp.ne.b32 p, %0, 0;\n" \
                " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n" \
                "}\n" :: \
                "r"((int)_pred), "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_));

// zfill ca type
#define CP_ASYNC_ZFILL_CA(_sm_v4, _gm_v4) \
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"(_16BYTE_));

#define CP_ASYNC_ZFILL_CA_L2_PREFETCH(_sm_v4, _gm_v4) \
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"(_16BYTE_));

// zfill cg type
#define CP_ASYNC_ZFILL_CG(_sm_v4, _gm_v4) \
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"(_16BYTE_));

#define CP_ASYNC_ZFILL_CG_L2_PREFETCH(_sm_v4, _gm_v4) \
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"(_16BYTE_));

// pred zfill ca type
#define PRED_CP_ASYNC_ZFILL_CA(_pred, _sm_v4, _gm_v4) \
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"((_pred) ? _16BYTE_ : _0BYTE_));

#define PRED_CP_ASYNC_ZFILL_CA_L2_PREFETCH(_pred, _sm_v4, _gm_v4) \
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"((_pred) ? _16BYTE_ : _0BYTE_));

// pred zfill cg type
#define PRED_CP_ASYNC_ZFILL_CG(_pred, _sm_v4, _gm_v4) \
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"((_pred) ? _16BYTE_ : _0BYTE_));

#define PRED_CP_ASYNC_ZFILL_CG_L2_PREFETCH(_pred, _sm_v4, _gm_v4) \
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" :: \
                "r"(_sm_v4), "l"(_gm_v4), "n"(_16BYTE_), "r"((_pred) ? _16BYTE_ : _0BYTE_));

////////////////////////////////////////
// fense macros
////////////////////////////////////////


#define CP_ASYNC_FENSE() \
        asm volatile("cp.async.commit_group;\n" ::);

#define CP_ASYNC_WAIT_ALL_BUT(_n) \
        asm volatile("cp.async.wait_group %0;\n" :: "n"(_n));

#define CP_ASYNC_WAIT_ALL() \
        asm volatile("cp.async.wait_all;\n" ::);

