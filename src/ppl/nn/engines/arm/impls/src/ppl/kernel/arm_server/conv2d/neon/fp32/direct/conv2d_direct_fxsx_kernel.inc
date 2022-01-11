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

#ifndef PPL_ARM_SK_PP_STR
#define PPL_ARM_SK_PP_STR(X) #X
#endif
#ifndef PPL_ARM_SK_STR
#define PPL_ARM_SK_STR(X) PPL_ARM_SK_PP_STR(X)
#endif

template<>
void ppl_arm_server_kernel_fp32_conv_direct_n4cx_h1wx_func<8, DST_TILE_W()>(
    const float *input_base,
    const float *filter_base,
    const float *bias_base,
          float *output_base,
          float *sum_base,
    int64_t       ic_tile_pck,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t flt_next_w_bytes,
    const int64_t flt_next_hw_bytes,
    const int64_t outh_x_outw_x_ocblk_bytes,
    const int64_t inh_x_inw_x_icblk_bytes,
    const int64_t dltnh_x_inw_x_icblk_bytes,
    const int64_t dltnw_x_icblk_bytes,
    const int64_t strdw_x_icblk_bytes,
    const uint32_t fuse_flag)
{
#define IF_OW_GT(OW, INSTRUCTION) ".if " PPL_ARM_SK_STR(DST_TILE_W()) " > " #OW "\n\t  " INSTRUCTION ".endif\n\t"

    // x9  : filter pointer
    // x10 : input pointer
    // x11 : output base pointer
    // x12 : output next 8-channel base pointer
    // x13 : iterator over filter height
    // x14 : iterator over filter width
    // x15 : 
    asm volatile (
        ".align 2\n\t"
        "cmp %[biasPtr], #0\n\t"
        "mov x9,  %[filterPtr]\n\t"
        "mov x10, %[inPtr]\n\t"

        "add x12, %[outPtr], %[outH_x_outW_x_ocV_bytes]\n\t"  // scheduled earlier
        "beq 0f\n\t"

        "ldp q10, q20, [%[biasPtr]]\n\t"
        IF_OW_GT(1, "mov v11.16b, v10.16b\n\t")
        IF_OW_GT(2, "mov v12.16b, v10.16b\n\t")
        IF_OW_GT(3, "mov v13.16b, v10.16b\n\t")
        IF_OW_GT(4, "mov v14.16b, v10.16b\n\t")
        IF_OW_GT(5, "mov v15.16b, v10.16b\n\t")
        IF_OW_GT(6, "mov v16.16b, v10.16b\n\t")
        IF_OW_GT(7, "mov v17.16b, v10.16b\n\t")
        IF_OW_GT(8, "mov v18.16b, v10.16b\n\t")
        IF_OW_GT(9, "mov v19.16b, v10.16b\n\t")

        IF_OW_GT(1, "mov v21.16b, v20.16b\n\t")
        IF_OW_GT(2, "mov v22.16b, v20.16b\n\t")
        IF_OW_GT(3, "mov v23.16b, v20.16b\n\t")
        IF_OW_GT(4, "mov v24.16b, v20.16b\n\t")
        IF_OW_GT(5, "mov v25.16b, v20.16b\n\t")
        IF_OW_GT(6, "mov v26.16b, v20.16b\n\t")
        IF_OW_GT(7, "mov v27.16b, v20.16b\n\t")
        IF_OW_GT(8, "mov v28.16b, v20.16b\n\t")
        IF_OW_GT(9, "mov v29.16b, v20.16b\n\t")
        "b 1f\n\t"

"0:\n\t" // RELOAD OUTPUT
                    "ldr q10, [%[outPtr]      ]\n\t"
        IF_OW_GT(1, "ldr q11, [%[outPtr], #16 ]\n\t")
        IF_OW_GT(2, "ldr q12, [%[outPtr], #32 ]\n\t")
        IF_OW_GT(3, "ldr q13, [%[outPtr], #48 ]\n\t")
        IF_OW_GT(4, "ldr q14, [%[outPtr], #64 ]\n\t")
        IF_OW_GT(5, "ldr q15, [%[outPtr], #80 ]\n\t")
        IF_OW_GT(6, "ldr q16, [%[outPtr], #96 ]\n\t")
        IF_OW_GT(7, "ldr q17, [%[outPtr], #112]\n\t")
        IF_OW_GT(8, "ldr q18, [%[outPtr], #128]\n\t")
        IF_OW_GT(9, "ldr q19, [%[outPtr], #144]\n\t")
                    "ldr q20, [x12      ]\n\t"
        IF_OW_GT(1, "ldr q21, [x12, #16 ]\n\t")
        IF_OW_GT(2, "ldr q22, [x12, #32 ]\n\t")
        IF_OW_GT(3, "ldr q23, [x12, #48 ]\n\t")
        IF_OW_GT(4, "ldr q24, [x12, #64 ]\n\t")
        IF_OW_GT(5, "ldr q25, [x12, #80 ]\n\t")
        IF_OW_GT(6, "ldr q26, [x12, #96 ]\n\t")
        IF_OW_GT(7, "ldr q27, [x12, #112]\n\t")
        IF_OW_GT(8, "ldr q28, [x12, #128]\n\t")
        IF_OW_GT(9, "ldr q29, [x12, #144]\n\t")

"1:\n\t"  // NANO KERNEL
"3:\n\t"  // LOOP OVER INPUT CHANNEL BY 4C
        "ldp q30, q31, [x9], #32\n\t"  // scheduled earlier
        "mov x13, %[fltH]\n\t"
"4:\n\t"  // LOOP OVER FILTER HEIGHT
        "mov x14, %[fltW]\n\t"

"5:\n\t"  // LOOP OVER FILTER WIDTH
        "subs x14, x14, #1\n\t"  // scheduled earlier

                    "ldr q0, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t"
        IF_OW_GT(1, "ldr q1, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(2, "ldr q2, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(3, "ldr q3, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(4, "ldr q4, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(5, "ldr q5, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(6, "ldr q6, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(7, "ldr q7, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(8, "ldr q8, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(9, "ldr q9, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
                                        "add x10, x10, %[diffW_x_icV_bytes]\n\t"

                    "fmla v10.4s, v30.4s, v0.s[0]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v30.4s, v1.s[0]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v30.4s, v2.s[0]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v30.4s, v3.s[0]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v30.4s, v4.s[0]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v30.4s, v5.s[0]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v30.4s, v6.s[0]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v30.4s, v7.s[0]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v30.4s, v8.s[0]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v30.4s, v9.s[0]\n\t")
        
        "ldr q30, [x9], #16\n\t"

                    "fmla v20.4s, v31.4s, v0.s[0]\n\t"
        IF_OW_GT(1, "fmla v21.4s, v31.4s, v1.s[0]\n\t")
        IF_OW_GT(2, "fmla v22.4s, v31.4s, v2.s[0]\n\t")
        IF_OW_GT(3, "fmla v23.4s, v31.4s, v3.s[0]\n\t")
        IF_OW_GT(4, "fmla v24.4s, v31.4s, v4.s[0]\n\t")
        IF_OW_GT(5, "fmla v25.4s, v31.4s, v5.s[0]\n\t")
        IF_OW_GT(6, "fmla v26.4s, v31.4s, v6.s[0]\n\t")
        IF_OW_GT(7, "fmla v27.4s, v31.4s, v7.s[0]\n\t")
        IF_OW_GT(8, "fmla v28.4s, v31.4s, v8.s[0]\n\t")
        IF_OW_GT(9, "fmla v29.4s, v31.4s, v9.s[0]\n\t")

        "ldr q31, [x9], #16\n\t"

                    "fmla v10.4s, v30.4s, v0.s[1]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v30.4s, v1.s[1]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v30.4s, v2.s[1]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v30.4s, v3.s[1]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v30.4s, v4.s[1]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v30.4s, v5.s[1]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v30.4s, v6.s[1]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v30.4s, v7.s[1]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v30.4s, v8.s[1]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v30.4s, v9.s[1]\n\t")
        
        "ldr q30, [x9], #16\n\t"

                    "fmla v20.4s, v31.4s, v0.s[1]\n\t"
        IF_OW_GT(1, "fmla v21.4s, v31.4s, v1.s[1]\n\t")
        IF_OW_GT(2, "fmla v22.4s, v31.4s, v2.s[1]\n\t")
        IF_OW_GT(3, "fmla v23.4s, v31.4s, v3.s[1]\n\t")
        IF_OW_GT(4, "fmla v24.4s, v31.4s, v4.s[1]\n\t")
        IF_OW_GT(5, "fmla v25.4s, v31.4s, v5.s[1]\n\t")
        IF_OW_GT(6, "fmla v26.4s, v31.4s, v6.s[1]\n\t")
        IF_OW_GT(7, "fmla v27.4s, v31.4s, v7.s[1]\n\t")
        IF_OW_GT(8, "fmla v28.4s, v31.4s, v8.s[1]\n\t")
        IF_OW_GT(9, "fmla v29.4s, v31.4s, v9.s[1]\n\t")

        "ldr q31, [x9], #16\n\t"

                    "fmla v10.4s, v30.4s, v0.s[2]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v30.4s, v1.s[2]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v30.4s, v2.s[2]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v30.4s, v3.s[2]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v30.4s, v4.s[2]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v30.4s, v5.s[2]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v30.4s, v6.s[2]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v30.4s, v7.s[2]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v30.4s, v8.s[2]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v30.4s, v9.s[2]\n\t")
        
        "ldr q30, [x9], #16\n\t"

                    "fmla v20.4s, v31.4s, v0.s[2]\n\t"
        IF_OW_GT(1, "fmla v21.4s, v31.4s, v1.s[2]\n\t")
        IF_OW_GT(2, "fmla v22.4s, v31.4s, v2.s[2]\n\t")
        IF_OW_GT(3, "fmla v23.4s, v31.4s, v3.s[2]\n\t")
        IF_OW_GT(4, "fmla v24.4s, v31.4s, v4.s[2]\n\t")
        IF_OW_GT(5, "fmla v25.4s, v31.4s, v5.s[2]\n\t")
        IF_OW_GT(6, "fmla v26.4s, v31.4s, v6.s[2]\n\t")
        IF_OW_GT(7, "fmla v27.4s, v31.4s, v7.s[2]\n\t")
        IF_OW_GT(8, "fmla v28.4s, v31.4s, v8.s[2]\n\t")
        IF_OW_GT(9, "fmla v29.4s, v31.4s, v9.s[2]\n\t")

        "ldr q31, [x9], #16\n\t"

                    "fmla v10.4s, v30.4s, v0.s[3]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v30.4s, v1.s[3]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v30.4s, v2.s[3]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v30.4s, v3.s[3]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v30.4s, v4.s[3]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v30.4s, v5.s[3]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v30.4s, v6.s[3]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v30.4s, v7.s[3]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v30.4s, v8.s[3]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v30.4s, v9.s[3]\n\t")
        
        "bgt 9f\n\t"
        "add x9,  x9,  %[fltDiffW_bytes]\n\t"
"9:\n\t"
        "ldr q30, [x9], #16\n\t"

                    "fmla v20.4s, v31.4s, v0.s[3]\n\t"
        IF_OW_GT(1, "fmla v21.4s, v31.4s, v1.s[3]\n\t")
        IF_OW_GT(2, "fmla v22.4s, v31.4s, v2.s[3]\n\t")
        IF_OW_GT(3, "fmla v23.4s, v31.4s, v3.s[3]\n\t")
        IF_OW_GT(4, "fmla v24.4s, v31.4s, v4.s[3]\n\t")
        IF_OW_GT(5, "fmla v25.4s, v31.4s, v5.s[3]\n\t")
        IF_OW_GT(6, "fmla v26.4s, v31.4s, v6.s[3]\n\t")
        IF_OW_GT(7, "fmla v27.4s, v31.4s, v7.s[3]\n\t")
        IF_OW_GT(8, "fmla v28.4s, v31.4s, v8.s[3]\n\t")
        IF_OW_GT(9, "fmla v29.4s, v31.4s, v9.s[3]\n\t")

        "ldr q31, [x9], #16\n\t"

        "bne 5b\n\t"
"2:\n\t"
        "subs x13, x13, #1\n\t"
        "add x10, x10, %[diffHW_x_icV_bytes]\n\t"
        "bne 4b\n\t"

        "subs %[icS], %[icS], #4\n\t"
        "add x9,  x9,  %[fltDiffHW_bytes]\n\t"
        "add x10, x10, %[inDiffH_x_inW_x_icV_bytes]\n\t"
        "bne 3b\n\t"

        "cmp %w[fuse_op], #0\n\t"
        "beq 8f\n\t"

        "ands w13, %w[fuse_op], #4\n\t"
        "beq 6f\n\t"

        "add x14, %[sumPtr], %[outH_x_outW_x_ocV_bytes]\n\t"
        // OC[0:3]
                    "ldr  q0, [%[sumPtr]      ]\n\t"
        IF_OW_GT(1, "ldr  q1, [%[sumPtr], #16 ]\n\t")
        IF_OW_GT(2, "ldr  q2, [%[sumPtr], #32 ]\n\t")
        IF_OW_GT(3, "ldr  q3, [%[sumPtr], #48 ]\n\t")
        IF_OW_GT(4, "ldr  q4, [%[sumPtr], #64 ]\n\t")
        IF_OW_GT(5, "ldr  q5, [%[sumPtr], #80 ]\n\t")
        IF_OW_GT(6, "ldr  q6, [%[sumPtr], #96 ]\n\t")
        IF_OW_GT(7, "ldr  q7, [%[sumPtr], #112]\n\t")
        IF_OW_GT(8, "ldr  q8, [%[sumPtr], #128]\n\t")
        IF_OW_GT(9, "ldr  q9, [%[sumPtr], #144]\n\t")
                    "fadd v10.4s, v10.4s, v0.4s\n\t"
        IF_OW_GT(1, "fadd v11.4s, v11.4s, v1.4s\n\t")
        IF_OW_GT(2, "fadd v12.4s, v12.4s, v2.4s\n\t")
        IF_OW_GT(3, "fadd v13.4s, v13.4s, v3.4s\n\t")
        IF_OW_GT(4, "fadd v14.4s, v14.4s, v4.4s\n\t")
        IF_OW_GT(5, "fadd v15.4s, v15.4s, v5.4s\n\t")
        IF_OW_GT(6, "fadd v16.4s, v16.4s, v6.4s\n\t")
        IF_OW_GT(7, "fadd v17.4s, v17.4s, v7.4s\n\t")
        IF_OW_GT(8, "fadd v18.4s, v18.4s, v8.4s\n\t")
        IF_OW_GT(9, "fadd v19.4s, v19.4s, v9.4s\n\t")
        // OC[4:7]
                    "ldr q0, [x14      ]\n\t"
        IF_OW_GT(1, "ldr q1, [x14, #16 ]\n\t")
        IF_OW_GT(2, "ldr q2, [x14, #32 ]\n\t")
        IF_OW_GT(3, "ldr q3, [x14, #48 ]\n\t")
        IF_OW_GT(4, "ldr q4, [x14, #64 ]\n\t")
        IF_OW_GT(5, "ldr q5, [x14, #80 ]\n\t")
        IF_OW_GT(6, "ldr q6, [x14, #96 ]\n\t")
        IF_OW_GT(7, "ldr q7, [x14, #112]\n\t")
        IF_OW_GT(8, "ldr q8, [x14, #128]\n\t")
        IF_OW_GT(9, "ldr q9, [x14, #144]\n\t")
                    "fadd v20.4s, v20.4s, v0.4s\n\t"
        IF_OW_GT(1, "fadd v21.4s, v21.4s, v1.4s\n\t")
        IF_OW_GT(2, "fadd v22.4s, v22.4s, v2.4s\n\t")
        IF_OW_GT(3, "fadd v23.4s, v23.4s, v3.4s\n\t")
        IF_OW_GT(4, "fadd v24.4s, v24.4s, v4.4s\n\t")
        IF_OW_GT(5, "fadd v25.4s, v25.4s, v5.4s\n\t")
        IF_OW_GT(6, "fadd v26.4s, v26.4s, v6.4s\n\t")
        IF_OW_GT(7, "fadd v27.4s, v27.4s, v7.4s\n\t")
        IF_OW_GT(8, "fadd v28.4s, v28.4s, v8.4s\n\t")
        IF_OW_GT(9, "fadd v29.4s, v29.4s, v9.4s\n\t")

"6:\n\t"
        "ands w13, %w[fuse_op], #2\n\t"
        "beq 7f\n\t"

        "fmov s6, 6.0e+0\n\t"
        "dup v6.4s, v6.s[0]\n\t"
                    "fmin v10.4s, v10.4s, v6.4s\n\t"
        IF_OW_GT(1, "fmin v11.4s, v11.4s, v6.4s\n\t")
        IF_OW_GT(2, "fmin v12.4s, v12.4s, v6.4s\n\t")
        IF_OW_GT(3, "fmin v13.4s, v13.4s, v6.4s\n\t")
        IF_OW_GT(4, "fmin v14.4s, v14.4s, v6.4s\n\t")
        IF_OW_GT(5, "fmin v15.4s, v15.4s, v6.4s\n\t")
        IF_OW_GT(6, "fmin v16.4s, v16.4s, v6.4s\n\t")
        IF_OW_GT(7, "fmin v17.4s, v17.4s, v6.4s\n\t")
        IF_OW_GT(8, "fmin v18.4s, v18.4s, v6.4s\n\t")
        IF_OW_GT(9, "fmin v19.4s, v19.4s, v6.4s\n\t")
                    "fmin v20.4s, v20.4s, v6.4s\n\t"
        IF_OW_GT(1, "fmin v21.4s, v21.4s, v6.4s\n\t")
        IF_OW_GT(2, "fmin v22.4s, v22.4s, v6.4s\n\t")
        IF_OW_GT(3, "fmin v23.4s, v23.4s, v6.4s\n\t")
        IF_OW_GT(4, "fmin v24.4s, v24.4s, v6.4s\n\t")
        IF_OW_GT(5, "fmin v25.4s, v25.4s, v6.4s\n\t")
        IF_OW_GT(6, "fmin v26.4s, v26.4s, v6.4s\n\t")
        IF_OW_GT(7, "fmin v27.4s, v27.4s, v6.4s\n\t")
        IF_OW_GT(8, "fmin v28.4s, v28.4s, v6.4s\n\t")
        IF_OW_GT(9, "fmin v29.4s, v29.4s, v6.4s\n\t")

"7:\n\t"
        "ands w13, %w[fuse_op], #1\n\t"
        "beq 8f\n\t"
        
        "eor v0.16b, v0.16b, v0.16b\n\t"
                    "fmax v10.4s, v10.4s, v0.4s\n\t"
        IF_OW_GT(1, "fmax v11.4s, v11.4s, v0.4s\n\t")
        IF_OW_GT(2, "fmax v12.4s, v12.4s, v0.4s\n\t")
        IF_OW_GT(3, "fmax v13.4s, v13.4s, v0.4s\n\t")
        IF_OW_GT(4, "fmax v14.4s, v14.4s, v0.4s\n\t")
        IF_OW_GT(5, "fmax v15.4s, v15.4s, v0.4s\n\t")
        IF_OW_GT(6, "fmax v16.4s, v16.4s, v0.4s\n\t")
        IF_OW_GT(7, "fmax v17.4s, v17.4s, v0.4s\n\t")
        IF_OW_GT(8, "fmax v18.4s, v18.4s, v0.4s\n\t")
        IF_OW_GT(9, "fmax v19.4s, v19.4s, v0.4s\n\t")
                    "fmax v20.4s, v20.4s, v0.4s\n\t"
        IF_OW_GT(1, "fmax v21.4s, v21.4s, v0.4s\n\t")
        IF_OW_GT(2, "fmax v22.4s, v22.4s, v0.4s\n\t")
        IF_OW_GT(3, "fmax v23.4s, v23.4s, v0.4s\n\t")
        IF_OW_GT(4, "fmax v24.4s, v24.4s, v0.4s\n\t")
        IF_OW_GT(5, "fmax v25.4s, v25.4s, v0.4s\n\t")
        IF_OW_GT(6, "fmax v26.4s, v26.4s, v0.4s\n\t")
        IF_OW_GT(7, "fmax v27.4s, v27.4s, v0.4s\n\t")
        IF_OW_GT(8, "fmax v28.4s, v28.4s, v0.4s\n\t")
        IF_OW_GT(9, "fmax v29.4s, v29.4s, v0.4s\n\t")

"8:\n\t"
        // Store OC[0:3]
                    "str q10, [%[outPtr]      ]\n\t"
        IF_OW_GT(1, "str q11, [%[outPtr], #16 ]\n\t")
        IF_OW_GT(2, "str q12, [%[outPtr], #32 ]\n\t")
        IF_OW_GT(3, "str q13, [%[outPtr], #48 ]\n\t")
        IF_OW_GT(4, "str q14, [%[outPtr], #64 ]\n\t")
        IF_OW_GT(5, "str q15, [%[outPtr], #80 ]\n\t")
        IF_OW_GT(6, "str q16, [%[outPtr], #96 ]\n\t")
        IF_OW_GT(7, "str q17, [%[outPtr], #112]\n\t")
        IF_OW_GT(8, "str q18, [%[outPtr], #128]\n\t")
        IF_OW_GT(9, "str q19, [%[outPtr], #144]\n\t")
        // Store OC[4:7]
                    "str q20, [x12      ]\n\t"
        IF_OW_GT(1, "str q21, [x12, #16 ]\n\t")
        IF_OW_GT(2, "str q22, [x12, #32 ]\n\t")
        IF_OW_GT(3, "str q23, [x12, #48 ]\n\t")
        IF_OW_GT(4, "str q24, [x12, #64 ]\n\t")
        IF_OW_GT(5, "str q25, [x12, #80 ]\n\t")
        IF_OW_GT(6, "str q26, [x12, #96 ]\n\t")
        IF_OW_GT(7, "str q27, [x12, #112]\n\t")
        IF_OW_GT(8, "str q28, [x12, #128]\n\t")
        IF_OW_GT(9, "str q29, [x12, #144]\n\t")
        :
        : 
          [inPtr]                       "r"  (input_base),
          [filterPtr]                   "r"  (filter_base),
          [biasPtr]                     "r"  (bias_base),
          [outPtr]                      "r"  (output_base),
          [sumPtr]                      "r"  (sum_base),
          [icS]                         "r"  (ic_tile_pck),
          [fltH]                        "r"  (flt_h),
          [fltW]                        "r"  (flt_w),
          [fltDiffW_bytes]              "r"  ((intptr_t)flt_next_w_bytes),
          [fltDiffHW_bytes]             "r"  ((intptr_t)flt_next_hw_bytes - 32),
          [outH_x_outW_x_ocV_bytes]     "r"  ((intptr_t)outh_x_outw_x_ocblk_bytes),
          [inDiffH_x_inW_x_icV_bytes]   "r"  ((intptr_t)inh_x_inw_x_icblk_bytes - (intptr_t)flt_h * (intptr_t)dltnh_x_inw_x_icblk_bytes),
          [diffHW_x_icV_bytes]          "r"  ((intptr_t)dltnh_x_inw_x_icblk_bytes - (intptr_t)flt_w * (intptr_t)dltnw_x_icblk_bytes),
          [diffW_x_icV_bytes]           "r"  ((intptr_t)dltnw_x_icblk_bytes - (intptr_t)(DST_TILE_W()) * (intptr_t)strdw_x_icblk_bytes),
          [strdW_x_icV_bytes]           "r"  ((intptr_t)strdw_x_icblk_bytes),
          [fuse_op]                     "r"  (fuse_flag)
        :
          "cc",
          "memory"
#ifndef __INTELLISENSE__
          ,
          "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          "v10" , "v11" , "v12" , "v13" , "v14" , "v15" , "v16" , "v17" ,
          "v18" , "v19" , "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v0", "v1", "v2", "v3",
          "v4", "v5", "v6", "v7", "v8", "v9", "v30", "v31"
#endif
    );
#undef IF_OW_GT
}


#ifndef PPL_ARM_SK_PP_STR
#define PPL_ARM_SK_PP_STR(X) #X
#endif
#ifndef PPL_ARM_SK_STR
#define PPL_ARM_SK_STR(X) PPL_ARM_SK_PP_STR(X)
#endif

template<>
void ppl_arm_server_kernel_fp32_conv_direct_n4cx_h1wx_func<4, DST_TILE_W()>(
    const float *input_base,
    const float *filter_base,
    const float *bias_base,
          float *output_base,
          float *sum_base,
    int64_t       ic_tile_pck,
    const int64_t flt_h,
    const int64_t flt_w,
    const int64_t flt_next_w_bytes,
    const int64_t flt_next_hw_bytes,
    const int64_t outh_x_outw_x_ocblk_bytes,
    const int64_t inh_x_inw_x_icblk_bytes,
    const int64_t dltnh_x_inw_x_icblk_bytes,
    const int64_t dltnw_x_icblk_bytes,
    const int64_t strdw_x_icblk_bytes,
    const uint32_t fuse_flag)
{
#define IF_OW_GT(OW, INSTRUCTION) ".if " PPL_ARM_SK_STR(DST_TILE_W()) " > " #OW "\n\t  " INSTRUCTION ".endif\n\t"

    // x9  : input 8-channel base pointer
    // x10 : input row base pointer
    // x11 : input col base pointer
    // x12 : 
    // x13 : iterator over filter height
    // x14 : iterator over filter width
    // x15 : 
    asm volatile (
        ".align 2\n\t"
        "cmp %[biasPtr], #0\n\t"
        "mov x9,  %[filterPtr]\n\t"
        "mov x10, %[inPtr]\n\t"
        "beq 0f\n\t"

        "ldr q10, [%[biasPtr]]\n\t"
        IF_OW_GT(1, "mov v11.16b, v10.16b\n\t")
        IF_OW_GT(2, "mov v12.16b, v10.16b\n\t")
        IF_OW_GT(3, "mov v13.16b, v10.16b\n\t")
        IF_OW_GT(4, "mov v14.16b, v10.16b\n\t")
        IF_OW_GT(5, "mov v15.16b, v10.16b\n\t")
        IF_OW_GT(6, "mov v16.16b, v10.16b\n\t")
        IF_OW_GT(7, "mov v17.16b, v10.16b\n\t")
        IF_OW_GT(8, "mov v18.16b, v10.16b\n\t")
        IF_OW_GT(9, "mov v19.16b, v10.16b\n\t")

        "b 1f\n\t"

"0:\n\t" // RELOAD OUTPUT
        "ldr q10, [%[outPtr]]\n\t"
        IF_OW_GT(1, "ldr q11, [%[outPtr], #16]\n\t")
        IF_OW_GT(2, "ldr q12, [%[outPtr], #32]\n\t")
        IF_OW_GT(3, "ldr q13, [%[outPtr], #48]\n\t")
        IF_OW_GT(4, "ldr q14, [%[outPtr], #64]\n\t")
        IF_OW_GT(5, "ldr q15, [%[outPtr], #80]\n\t")
        IF_OW_GT(6, "ldr q16, [%[outPtr], #96]\n\t")
        IF_OW_GT(7, "ldr q17, [%[outPtr], #112]\n\t")
        IF_OW_GT(8, "ldr q18, [%[outPtr], #128]\n\t")
        IF_OW_GT(9, "ldr q19, [%[outPtr], #144]\n\t")

"1:\n\t"  // NANO KERNEL
"3:\n\t"  // LOOP OVER INPUT CHANNEL BY 4C
        "ldr q30, [x9], #32\n\t"  // scheduled earlier
        "ldr q31, [x9], #32\n\t"  // scheduled earlier
        "mov x13, %[fltH]\n\t"
"4:\n\t"  // LOOP OVER FILTER HEIGHT
        "mov x14, %[fltW]\n\t"

"5:\n\t"  // LOOP OVER FILTER WIDTH
        "subs x14, x14, #1\n\t"  // scheduled earlier
        
                    "ldr q0, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t"
        IF_OW_GT(1, "ldr q1, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(2, "ldr q2, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(3, "ldr q3, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(4, "ldr q4, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(5, "ldr q5, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(6, "ldr q6, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(7, "ldr q7, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(8, "ldr q8, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
        IF_OW_GT(9, "ldr q9, [x10]\n\t" "add x10, x10, %[strdW_x_icV_bytes]\n\t")
                                        "add x10, x10, %[diffW_x_icV_bytes]\n\t"

                    "fmla v10.4s, v30.4s, v0.s[0]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v30.4s, v1.s[0]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v30.4s, v2.s[0]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v30.4s, v3.s[0]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v30.4s, v4.s[0]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v30.4s, v5.s[0]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v30.4s, v6.s[0]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v30.4s, v7.s[0]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v30.4s, v8.s[0]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v30.4s, v9.s[0]\n\t")
        
        "ldr q30, [x9], #32\n\t"

                    "fmla v10.4s, v31.4s, v0.s[1]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v31.4s, v1.s[1]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v31.4s, v2.s[1]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v31.4s, v3.s[1]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v31.4s, v4.s[1]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v31.4s, v5.s[1]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v31.4s, v6.s[1]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v31.4s, v7.s[1]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v31.4s, v8.s[1]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v31.4s, v9.s[1]\n\t")
        
        "ldr q31, [x9], #32\n\t"

                    "fmla v10.4s, v30.4s, v0.s[2]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v30.4s, v1.s[2]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v30.4s, v2.s[2]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v30.4s, v3.s[2]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v30.4s, v4.s[2]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v30.4s, v5.s[2]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v30.4s, v6.s[2]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v30.4s, v7.s[2]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v30.4s, v8.s[2]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v30.4s, v9.s[2]\n\t")

        "bgt 9f\n\t"
        "add x9,  x9,  %[fltDiffW_bytes]\n\t"
"9:\n\t"        
        "ldr q30, [x9], #32\n\t"

                    "fmla v10.4s, v31.4s, v0.s[3]\n\t"
        IF_OW_GT(1, "fmla v11.4s, v31.4s, v1.s[3]\n\t")
        IF_OW_GT(2, "fmla v12.4s, v31.4s, v2.s[3]\n\t")
        IF_OW_GT(3, "fmla v13.4s, v31.4s, v3.s[3]\n\t")
        IF_OW_GT(4, "fmla v14.4s, v31.4s, v4.s[3]\n\t")
        IF_OW_GT(5, "fmla v15.4s, v31.4s, v5.s[3]\n\t")
        IF_OW_GT(6, "fmla v16.4s, v31.4s, v6.s[3]\n\t")
        IF_OW_GT(7, "fmla v17.4s, v31.4s, v7.s[3]\n\t")
        IF_OW_GT(8, "fmla v18.4s, v31.4s, v8.s[3]\n\t")
        IF_OW_GT(9, "fmla v19.4s, v31.4s, v9.s[3]\n\t")
        
        "ldr q31, [x9], #32\n\t"

        "bne 5b\n\t"
"2:\n\t"
        "subs x13, x13, #1\n\t"
        "add x10, x10, %[diffHW_x_icV_bytes]\n\t"
        "bne 4b\n\t"

        "subs %[icS], %[icS], #4\n\t"
        "add x9,  x9,  %[fltDiffHW_bytes]\n\t"
        "add x10, x10, %[inDiffH_x_inW_x_icV_bytes]\n\t"
        "bne 3b\n\t"

        "cmp %w[fuse_op], #0\n\t"
        "beq 8f\n\t"

        "ands w13, %w[fuse_op], #4\n\t"
        "beq 6f\n\t"

        "add x14, %[sumPtr], %[outH_x_outW_x_ocV_bytes]\n\t"
        // OC[0:3]
                    "ldr  q0, [%[sumPtr]      ]\n\t"
        IF_OW_GT(1, "ldr  q1, [%[sumPtr], #16 ]\n\t")
        IF_OW_GT(2, "ldr  q2, [%[sumPtr], #32 ]\n\t")
        IF_OW_GT(3, "ldr  q3, [%[sumPtr], #48 ]\n\t")
        IF_OW_GT(4, "ldr  q4, [%[sumPtr], #64 ]\n\t")
        IF_OW_GT(5, "ldr  q5, [%[sumPtr], #80 ]\n\t")
        IF_OW_GT(6, "ldr  q6, [%[sumPtr], #96 ]\n\t")
        IF_OW_GT(7, "ldr  q7, [%[sumPtr], #112]\n\t")
        IF_OW_GT(8, "ldr  q8, [%[sumPtr], #128]\n\t")
        IF_OW_GT(9, "ldr  q9, [%[sumPtr], #144]\n\t")
                    "fadd v10.4s, v10.4s, v0.4s\n\t"
        IF_OW_GT(1, "fadd v11.4s, v11.4s, v1.4s\n\t")
        IF_OW_GT(2, "fadd v12.4s, v12.4s, v2.4s\n\t")
        IF_OW_GT(3, "fadd v13.4s, v13.4s, v3.4s\n\t")
        IF_OW_GT(4, "fadd v14.4s, v14.4s, v4.4s\n\t")
        IF_OW_GT(5, "fadd v15.4s, v15.4s, v5.4s\n\t")
        IF_OW_GT(6, "fadd v16.4s, v16.4s, v6.4s\n\t")
        IF_OW_GT(7, "fadd v17.4s, v17.4s, v7.4s\n\t")
        IF_OW_GT(8, "fadd v18.4s, v18.4s, v8.4s\n\t")
        IF_OW_GT(9, "fadd v19.4s, v19.4s, v9.4s\n\t")

"6:\n\t"
        "ands w13, %w[fuse_op], #2\n\t"
        "beq 7f\n\t"

        "fmov s6, 6.0e+0\n\t"
        "dup v6.4s, v6.s[0]\n\t"
                    "fmin v10.4s, v10.4s, v6.4s\n\t"
        IF_OW_GT(1, "fmin v11.4s, v11.4s, v6.4s\n\t")
        IF_OW_GT(2, "fmin v12.4s, v12.4s, v6.4s\n\t")
        IF_OW_GT(3, "fmin v13.4s, v13.4s, v6.4s\n\t")
        IF_OW_GT(4, "fmin v14.4s, v14.4s, v6.4s\n\t")
        IF_OW_GT(5, "fmin v15.4s, v15.4s, v6.4s\n\t")
        IF_OW_GT(6, "fmin v16.4s, v16.4s, v6.4s\n\t")
        IF_OW_GT(7, "fmin v17.4s, v17.4s, v6.4s\n\t")
        IF_OW_GT(8, "fmin v18.4s, v18.4s, v6.4s\n\t")
        IF_OW_GT(9, "fmin v19.4s, v19.4s, v6.4s\n\t")

"7:\n\t"
        "ands w13, %w[fuse_op], #1\n\t"
        "beq 8f\n\t"
        
        "eor v0.16b, v0.16b, v0.16b\n\t"
                    "fmax v10.4s, v10.4s, v0.4s\n\t"
        IF_OW_GT(1, "fmax v11.4s, v11.4s, v0.4s\n\t")
        IF_OW_GT(2, "fmax v12.4s, v12.4s, v0.4s\n\t")
        IF_OW_GT(3, "fmax v13.4s, v13.4s, v0.4s\n\t")
        IF_OW_GT(4, "fmax v14.4s, v14.4s, v0.4s\n\t")
        IF_OW_GT(5, "fmax v15.4s, v15.4s, v0.4s\n\t")
        IF_OW_GT(6, "fmax v16.4s, v16.4s, v0.4s\n\t")
        IF_OW_GT(7, "fmax v17.4s, v17.4s, v0.4s\n\t")
        IF_OW_GT(8, "fmax v18.4s, v18.4s, v0.4s\n\t")
        IF_OW_GT(9, "fmax v19.4s, v19.4s, v0.4s\n\t")

"8:\n\t"
        // Store OC[0:3]
                    "str q10, [%[outPtr]      ]\n\t"
        IF_OW_GT(1, "str q11, [%[outPtr], #16 ]\n\t")
        IF_OW_GT(2, "str q12, [%[outPtr], #32 ]\n\t")
        IF_OW_GT(3, "str q13, [%[outPtr], #48 ]\n\t")
        IF_OW_GT(4, "str q14, [%[outPtr], #64 ]\n\t")
        IF_OW_GT(5, "str q15, [%[outPtr], #80 ]\n\t")
        IF_OW_GT(6, "str q16, [%[outPtr], #96 ]\n\t")
        IF_OW_GT(7, "str q17, [%[outPtr], #112]\n\t")
        IF_OW_GT(8, "str q18, [%[outPtr], #128]\n\t")
        IF_OW_GT(9, "str q19, [%[outPtr], #144]\n\t")
        :
        : 
          [inPtr]                       "r"  (input_base),
          [filterPtr]                   "r"  (filter_base),
          [biasPtr]                     "r"  (bias_base),
          [outPtr]                      "r"  (output_base),
          [sumPtr]                      "r"  (sum_base),
          [icS]                         "r"  (ic_tile_pck),
          [fltH]                        "r"  (flt_h),
          [fltW]                        "r"  (flt_w),
          [fltDiffW_bytes]              "r"  ((intptr_t)flt_next_w_bytes),
          [fltDiffHW_bytes]             "r"  ((intptr_t)flt_next_hw_bytes - 64),
          [outH_x_outW_x_ocV_bytes]     "r"  ((intptr_t)outh_x_outw_x_ocblk_bytes),
          [inDiffH_x_inW_x_icV_bytes]   "r"  ((intptr_t)inh_x_inw_x_icblk_bytes - (intptr_t)flt_h * (intptr_t)dltnh_x_inw_x_icblk_bytes),
          [diffHW_x_icV_bytes]          "r"  ((intptr_t)dltnh_x_inw_x_icblk_bytes - (intptr_t)flt_w * (intptr_t)dltnw_x_icblk_bytes),
          [diffW_x_icV_bytes]           "r"  ((intptr_t)dltnw_x_icblk_bytes - (intptr_t)DST_TILE_W() * (intptr_t)strdw_x_icblk_bytes),
          [strdW_x_icV_bytes]           "r"  ((intptr_t)strdw_x_icblk_bytes),
          [fuse_op]                     "r"  (fuse_flag)
        :
          "cc",
          "memory"
#ifndef __INTELLISENSE__
          ,
          "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          "v10" , "v11" , "v12" , "v13" , "v14" , "v15" , "v16" , "v17" ,
          "v18" , "v19" , "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v0", "v1", "v2", "v3",
          "v4", "v5", "v6", "v7", "v8", "v9", "v30", "v31"
#endif
    );
#undef IF_OW_GT
}
