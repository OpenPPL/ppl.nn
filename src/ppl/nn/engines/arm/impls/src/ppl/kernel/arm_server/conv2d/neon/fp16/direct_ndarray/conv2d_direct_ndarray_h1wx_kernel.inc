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
void ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_kernel<16, OUT_TILE_W()>(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    const int64_t inH,
    const int64_t inW,
    const int64_t inC,
    const int64_t fltH_valid,
    const int64_t fltW,
    const int64_t strdW,
    const int64_t dltnH,
    const int64_t dltnW,
    const int64_t filter_ic_stride,
    const int64_t outBCHW_stride,
    const uint32_t fuse_type)
{
#define IF_OW_GT(OW, INSTRUCTION) ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " > " #OW "\n\t  " INSTRUCTION ".endif\n\t"

    asm volatile (
        ".align 2\n\t"
        
        "cmp %[biasPtr], #0\n\t"
        "bne 0f\n\t"

        "add x12, %[outPtr], %[out_bchw_stride_bytes]\n\t"
        IF_OW_GT(0,  "ldr q2,  [%[outPtr]      ]\n\t")
        IF_OW_GT(1,  "ldr q3,  [%[outPtr], #16 ]\n\t")
        IF_OW_GT(2,  "ldr q4,  [%[outPtr], #32 ]\n\t")
        IF_OW_GT(3,  "ldr q5,  [%[outPtr], #48 ]\n\t")
        IF_OW_GT(4,  "ldr q6,  [%[outPtr], #64 ]\n\t")
        IF_OW_GT(5,  "ldr q7,  [%[outPtr], #80 ]\n\t")
        IF_OW_GT(6,  "ldr q8,  [%[outPtr], #96 ]\n\t")
        IF_OW_GT(7,  "ldr q9,  [%[outPtr], #112]\n\t")
        IF_OW_GT(8,  "ldr q10, [%[outPtr], #128]\n\t")
        IF_OW_GT(9,  "ldr q11, [%[outPtr], #144]\n\t")
        IF_OW_GT(10, "ldr q12, [%[outPtr], #160]\n\t")
        IF_OW_GT(11, "ldr q13, [%[outPtr], #176]\n\t")
        IF_OW_GT(12, "ldr q14, [%[outPtr], #192]\n\t")
        IF_OW_GT(13, "ldr q15, [%[outPtr], #208]\n\t")

        IF_OW_GT(0,  "ldr q16, [x12      ]\n\t")
        IF_OW_GT(1,  "ldr q17, [x12, #16 ]\n\t")
        IF_OW_GT(2,  "ldr q18, [x12, #32 ]\n\t")
        IF_OW_GT(3,  "ldr q19, [x12, #48 ]\n\t")
        IF_OW_GT(4,  "ldr q20, [x12, #64 ]\n\t")
        IF_OW_GT(5,  "ldr q21, [x12, #80 ]\n\t")
        IF_OW_GT(6,  "ldr q22, [x12, #96 ]\n\t")
        IF_OW_GT(7,  "ldr q23, [x12, #112]\n\t")
        IF_OW_GT(8,  "ldr q24, [x12, #128]\n\t")
        IF_OW_GT(9,  "ldr q25, [x12, #144]\n\t")
        IF_OW_GT(10, "ldr q26, [x12, #160]\n\t")
        IF_OW_GT(11, "ldr q27, [x12, #176]\n\t")
        IF_OW_GT(12, "ldr q28, [x12, #192]\n\t")
        IF_OW_GT(13, "ldr q29, [x12, #208]\n\t")

        "b 1f\n\t"

    "0:\n\t"
        "ld1 {v2.8h, v3.8h}, [%[biasPtr]]\n\t"
        "mov v16.16b, v3.16b\n\t"

        IF_OW_GT(1,  "mov v3.16b,  v2.16b\n\t")
        IF_OW_GT(2,  "mov v4.16b,  v2.16b\n\t")
        IF_OW_GT(3,  "mov v5.16b,  v2.16b\n\t")
        IF_OW_GT(4,  "mov v6.16b,  v2.16b\n\t")
        IF_OW_GT(5,  "mov v7.16b,  v2.16b\n\t")
        IF_OW_GT(6,  "mov v8.16b,  v2.16b\n\t")
        IF_OW_GT(7,  "mov v9.16b,  v2.16b\n\t")
        IF_OW_GT(8,  "mov v10.16b, v2.16b\n\t")
        IF_OW_GT(9,  "mov v11.16b, v2.16b\n\t")
        IF_OW_GT(10, "mov v12.16b, v2.16b\n\t")
        IF_OW_GT(11, "mov v13.16b, v2.16b\n\t")
        IF_OW_GT(12, "mov v14.16b, v2.16b\n\t")
        IF_OW_GT(13, "mov v15.16b, v2.16b\n\t")

        IF_OW_GT(1,  "mov v17.16b, v16.16b\n\t")
        IF_OW_GT(2,  "mov v18.16b, v16.16b\n\t")
        IF_OW_GT(3,  "mov v19.16b, v16.16b\n\t")
        IF_OW_GT(4,  "mov v20.16b, v16.16b\n\t")
        IF_OW_GT(5,  "mov v21.16b, v16.16b\n\t")
        IF_OW_GT(6,  "mov v22.16b, v16.16b\n\t")
        IF_OW_GT(7,  "mov v23.16b, v16.16b\n\t")
        IF_OW_GT(8,  "mov v24.16b, v16.16b\n\t")
        IF_OW_GT(9,  "mov v25.16b, v16.16b\n\t")
        IF_OW_GT(10, "mov v26.16b, v16.16b\n\t")
        IF_OW_GT(11, "mov v27.16b, v16.16b\n\t")
        IF_OW_GT(12, "mov v28.16b, v16.16b\n\t")
        IF_OW_GT(13, "mov v29.16b, v16.16b\n\t")

    "1:\n\t"
        "mov x9, %[flt_ptr]\n\t"
        "mov x10, %[in_ptr]\n\t"
        "mov x17, %[ic]\n\t"

        "cmp %[w_strd_bytes], #2\n\t"
        "bne 3f\n\t"

    "2:\n\t"
    "22:\n\t"
        "mov x11, x9\n\t"
        "add x9, x9, %[flt_ic_stride_bytes]\n\t"
        "mov x12, x10\n\t"
        "add x10, x10, %[hw_in_bytes]\n\t"
        "mov x13, %[flt_h_valid]\n\t"
    "23:\n\t"
        "mov x14, x12\n\t"
        "add x12, x12, %[h_dltn_w_in_bytes]\n\t"
        "mov x15, %[flt_w]\n\t"
    "24:\n\t"
        "subs x15, x15, #1\n\t"
        "mov x16, x14\n\t"
        "add x14, x14, %[w_dltn_bytes]\n\t"

        "ld1 {v30.8h, v31.8h}, [x11], #32\n\t"

        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 1\n\t"
            "ld1 {v0.h}[0], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " >= 2\n\t"
            ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " <= 4\n\t"
                "ld1 {v0.4h}, [x16]\n\t"
            ".endif\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " > 4\n\t"
            ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " <= 8\n\t"
                "ld1 {v0.8h}, [x16]\n\t"
            ".endif\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 9\n\t"
            "ld1 {v0.8h}, [x16], #16\n\t"
            "ld1 {v1.h}[0], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " >= 10\n\t"
            ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " <= 12\n\t"
                "ld1 {v0.8h}, [x16], #16\n\t"
                "ld1 {v1.4h}, [x16]\n\t"
            ".endif\n\t"
        ".endif\n\t"
        IF_OW_GT(12, "ld1 {v0.8h, v1.8h}, [x16]\n\t")
        
        IF_OW_GT(0,  "fmla v2.8h,  v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(0,  "fmla v16.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(1,  "fmla v3.8h,  v30.8h, v0.h[1]\n\t" )
        IF_OW_GT(1,  "fmla v17.8h, v31.8h, v0.h[1]\n\t" )

        IF_OW_GT(2,  "fmla v4.8h,  v30.8h, v0.h[2]\n\t" )
        IF_OW_GT(2,  "fmla v18.8h, v31.8h, v0.h[2]\n\t" )

        IF_OW_GT(3,  "fmla v5.8h,  v30.8h, v0.h[3]\n\t" )
        IF_OW_GT(3,  "fmla v19.8h, v31.8h, v0.h[3]\n\t" )

        IF_OW_GT(4,  "fmla v6.8h,  v30.8h, v0.h[4]\n\t" )
        IF_OW_GT(4,  "fmla v20.8h, v31.8h, v0.h[4]\n\t" )

        IF_OW_GT(5,  "fmla v7.8h,  v30.8h, v0.h[5]\n\t" )
        IF_OW_GT(5,  "fmla v21.8h, v31.8h, v0.h[5]\n\t" )

        IF_OW_GT(6,  "fmla v8.8h,  v30.8h, v0.h[6]\n\t" )
        IF_OW_GT(6,  "fmla v22.8h, v31.8h, v0.h[6]\n\t" )

        IF_OW_GT(7,  "fmla v9.8h,  v30.8h, v0.h[7]\n\t" )
        IF_OW_GT(7,  "fmla v23.8h, v31.8h, v0.h[7]\n\t" )

        IF_OW_GT(8,  "fmla v10.8h, v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(8,  "fmla v24.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(9,  "fmla v11.8h, v30.8h, v1.h[1]\n\t" )
        IF_OW_GT(9,  "fmla v25.8h, v31.8h, v1.h[1]\n\t" )

        IF_OW_GT(10, "fmla v12.8h, v30.8h, v1.h[2]\n\t" )
        IF_OW_GT(10, "fmla v26.8h, v31.8h, v1.h[2]\n\t" )

        IF_OW_GT(11, "fmla v13.8h, v30.8h, v1.h[3]\n\t" )
        IF_OW_GT(11, "fmla v27.8h, v31.8h, v1.h[3]\n\t" )

        IF_OW_GT(12, "fmla v14.8h, v30.8h, v1.h[4]\n\t" )
        IF_OW_GT(12, "fmla v28.8h, v31.8h, v1.h[4]\n\t" )

        IF_OW_GT(13, "fmla v15.8h, v30.8h, v1.h[5]\n\t" )
        IF_OW_GT(13, "fmla v29.8h, v31.8h, v1.h[5]\n\t" )
        
        "bgt 24b\n\t"
        
        "subs x13, x13, #1\n\t"
        "bgt 23b\n\t"

        "subs x17, x17, #1\n\t"
        "bgt 22b\n\t"
        "b 5f\n\t"

    "3:\n\t"
        "cmp %[w_strd_bytes], #4\n\t"
        "bne 4f\n\t"
    "32:\n\t"
        "mov x11, x9\n\t"
        "add x9, x9, %[flt_ic_stride_bytes]\n\t"
        "mov x12, x10\n\t"
        "add x10, x10, %[hw_in_bytes]\n\t"
        "mov x13, %[flt_h_valid]\n\t"
    "33:\n\t"
        "mov x14, x12\n\t"
        "add x12, x12, %[h_dltn_w_in_bytes]\n\t"
        "mov x15, %[flt_w]\n\t"
    "34:\n\t"
        "subs x15, x15, #1\n\t"
        "mov x16, x14\n\t"
        "add x14, x14, %[w_dltn_bytes]\n\t"

        "ld1 {v30.8h, v31.8h}, [x11], #32\n\t"

        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 1\n\t"
            "ld1 {v0.h}[0], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 2\n\t"
            "ld1 {v0.h}[0], [x16]\n\t"
            "add x16, x16, #4\n\t"
            "ld1 {v0.h}[2], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 3\n\t"
            "ld1 {v0.4h}, [x16], #8\n\t"
            "ld1 {v0.h}[4], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 4\n\t"
            "ld1 {v0.4h}, [x16], #8\n\t"
            "ld1 {v0.h}[4], [x16]\n\t"
            "add x16, x16, #4\n\t"
            "ld1 {v0.h}[6], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 5\n\t"
            "ld1 {v0.8h}, [x16], #16\n\t"
            "ld1 {v1.h}[0], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 6\n\t"
            "ld1 {v0.8h}, [x16], #16\n\t"
            "ld1 {v1.h}[0], [x16]\n\t"
            "add x16, x16, #4\n\t"
            "ld1 {v1.h}[2], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 7\n\t"
            "ld1 {v0.8h}, [x16], #16\n\t"
            "ld1 {v1.4h}, [x16], #8\n\t"
            "ld1 {v1.h}[4], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 8\n\t"
            "ld1 {v0.8h}, [x16], #16\n\t"
            "ld1 {v1.4h}, [x16], #8\n\t"
            "ld1 {v1.h}[4], [x16]\n\t"
            "add x16, x16, #4\n\t"
            "ld1 {v1.h}[6], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " > 8\n\t"
            "ld1 {v0.8h}, [x16], #16\n\t"
            "ld1 {v1.8h}, [x16], #16\n\t"
        ".endif\n\t"
        
        IF_OW_GT(0,  "fmla v2.8h,  v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(0,  "fmla v16.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(1,  "fmla v3.8h,  v30.8h, v0.h[2]\n\t" )
        IF_OW_GT(1,  "fmla v17.8h, v31.8h, v0.h[2]\n\t" )

        IF_OW_GT(2,  "fmla v4.8h,  v30.8h, v0.h[4]\n\t" )
        IF_OW_GT(2,  "fmla v18.8h, v31.8h, v0.h[4]\n\t" )

        IF_OW_GT(3,  "fmla v5.8h,  v30.8h, v0.h[6]\n\t" )
        IF_OW_GT(3,  "fmla v19.8h, v31.8h, v0.h[6]\n\t" )

        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 9\n\t"
            "ld1 {v0.h}[0], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 10\n\t"
            "ld1 {v0.h}[0], [x16]\n\t"
            "add x16, x16, #4\n\t"
            "ld1 {v0.h}[2], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 11\n\t"
            "ld1 {v0.4h}, [x16], #8\n\t"
            "ld1 {v0.h}[4], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 12\n\t"
            "ld1 {v0.4h}, [x16], #8\n\t"
            "ld1 {v0.h}[4], [x16]\n\t"
            "add x16, x16, #4\n\t"
            "ld1 {v0.h}[6], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " >= 13\n\t"
            "ld1 {v0.8h}, [x16], #16\n\t"
        ".endif\n\t"

        IF_OW_GT(4,  "fmla v6.8h,  v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(4,  "fmla v20.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(5,  "fmla v7.8h,  v30.8h, v1.h[2]\n\t" )
        IF_OW_GT(5,  "fmla v21.8h, v31.8h, v1.h[2]\n\t" )

        IF_OW_GT(6,  "fmla v8.8h,  v30.8h, v1.h[4]\n\t" )
        IF_OW_GT(6,  "fmla v22.8h, v31.8h, v1.h[4]\n\t" )

        IF_OW_GT(7,  "fmla v9.8h,  v30.8h, v1.h[6]\n\t" )
        IF_OW_GT(7,  "fmla v23.8h, v31.8h, v1.h[6]\n\t" )

        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " == 13\n\t"
            "ld1 {v1.h}[0], [x16]\n\t"
        ".endif\n\t"
        ".if " PPL_ARM_SK_STR(OUT_TILE_W()) " > 13\n\t"
            "ld1 {v1.h}[0], [x16]\n\t"
            "add x16, x16, #4\n\t"
            "ld1 {v1.h}[2], [x16]\n\t"
        ".endif\n\t"

        IF_OW_GT(8,  "fmla v10.8h, v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(8,  "fmla v24.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(9,  "fmla v11.8h, v30.8h, v0.h[2]\n\t" )
        IF_OW_GT(9,  "fmla v25.8h, v31.8h, v0.h[2]\n\t" )

        IF_OW_GT(10, "fmla v12.8h, v30.8h, v0.h[4]\n\t" )
        IF_OW_GT(10, "fmla v26.8h, v31.8h, v0.h[4]\n\t" )

        IF_OW_GT(11, "fmla v13.8h, v30.8h, v0.h[6]\n\t" )
        IF_OW_GT(11, "fmla v27.8h, v31.8h, v0.h[6]\n\t" )

        IF_OW_GT(12, "fmla v14.8h, v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(12, "fmla v28.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(13, "fmla v15.8h, v30.8h, v1.h[2]\n\t" )
        IF_OW_GT(13, "fmla v29.8h, v31.8h, v1.h[2]\n\t" )
        
        "bgt 34b\n\t"
        
        "subs x13, x13, #1\n\t"
        "bgt 33b\n\t"

        "subs x17, x17, #1\n\t"
        "bgt 32b\n\t"
        "b 5f\n\t"

    "4:\n\t"  
    "42:\n\t"
        "mov x11, x9\n\t"
        "add x9, x9, %[flt_ic_stride_bytes]\n\t"
        "mov x12, x10\n\t"
        "add x10, x10, %[hw_in_bytes]\n\t"
        "mov x13, %[flt_h_valid]\n\t"
    "43:\n\t"
        "mov x14, x12\n\t"
        "add x12, x12, %[h_dltn_w_in_bytes]\n\t"
        "mov x15, %[flt_w]\n\t"
    "44:\n\t"
        "subs x15, x15, #1\n\t"
        "mov x16, x14\n\t"
        "add x14, x14, %[w_dltn_bytes]\n\t"

        "ld1 {v30.8h, v31.8h}, [x11], #32\n\t"

        IF_OW_GT(0,  "ld1 {v0.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(0,  "fmla v2.8h,  v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(0,  "fmla v16.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(1,  "ld1 {v1.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(1,  "fmla v3.8h,  v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(1,  "fmla v17.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(2,  "ld1 {v0.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(2,  "fmla v4.8h,  v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(2,  "fmla v18.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(3,  "ld1 {v1.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(3,  "fmla v5.8h,  v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(3,  "fmla v19.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(4,  "ld1 {v0.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(4,  "fmla v6.8h,  v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(4,  "fmla v20.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(5,  "ld1 {v1.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(5,  "fmla v7.8h,  v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(5,  "fmla v21.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(6,  "ld1 {v0.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(6,  "fmla v8.8h,  v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(6,  "fmla v22.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(7,  "ld1 {v1.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(7,  "fmla v9.8h,  v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(7,  "fmla v23.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(8,  "ld1 {v0.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(8,  "fmla v10.8h, v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(8,  "fmla v24.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(9,  "ld1 {v1.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(9,  "fmla v11.8h, v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(9,  "fmla v25.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(10, "ld1 {v0.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(10, "fmla v12.8h, v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(10, "fmla v26.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(11, "ld1 {v1.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(11, "fmla v13.8h, v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(11, "fmla v27.8h, v31.8h, v1.h[0]\n\t" )

        IF_OW_GT(12, "ld1 {v0.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(12, "fmla v14.8h, v30.8h, v0.h[0]\n\t" )
        IF_OW_GT(12, "fmla v28.8h, v31.8h, v0.h[0]\n\t" )

        IF_OW_GT(13, "ld1 {v1.h}[0], [x16], %[w_strd_bytes]\n\t"   )
        IF_OW_GT(13, "fmla v15.8h, v30.8h, v1.h[0]\n\t" )
        IF_OW_GT(13, "fmla v29.8h, v31.8h, v1.h[0]\n\t" )
        
        "bgt 44b\n\t"
        
        "subs x13, x13, #1\n\t"
        "bgt 43b\n\t"

        "subs x17, x17, #1\n\t"
        "bgt 42b\n\t"

    "5:\n\t"
        "ands w13, %w[fuse_op], #4\n\t"
        "beq 6f\n\t"

        "add x12, %[add_ptr], %[out_bchw_stride_bytes]\n\t"

        IF_OW_GT(0 , "ldr q0, [%[add_ptr]      ]\n\t"  )
        IF_OW_GT(0 , "add v2.8h, v2.8h, v0.8h\n\t"     )
        IF_OW_GT(1 , "ldr q1, [%[add_ptr], #16 ]\n\t"  )
        IF_OW_GT(1 , "add v3.8h, v3.8h, v1.8h\n\t"     )
        IF_OW_GT(2 , "ldr q0, [%[add_ptr], #32 ]\n\t"  )
        IF_OW_GT(2 , "add v4.8h, v4.8h, v0.8h\n\t"     )
        IF_OW_GT(3 , "ldr q1, [%[add_ptr], #48 ]\n\t"  )
        IF_OW_GT(3 , "add v5.8h, v5.8h, v1.8h\n\t"     )
        IF_OW_GT(4 , "ldr q0, [%[add_ptr], #64 ]\n\t"  )
        IF_OW_GT(4 , "add v6.8h, v6.8h, v0.8h\n\t"     )
        IF_OW_GT(5 , "ldr q1, [%[add_ptr], #80 ]\n\t"  )
        IF_OW_GT(5 , "add v7.8h, v7.8h, v1.8h\n\t"     )
        IF_OW_GT(6 , "ldr q0, [%[add_ptr], #96 ]\n\t"  )
        IF_OW_GT(6 , "add v8.8h, v8.8h, v0.8h\n\t"     )
        IF_OW_GT(7 , "ldr q1, [%[add_ptr], #112]\n\t"  )
        IF_OW_GT(7 , "add v9.8h, v9.8h, v1.8h\n\t"     )
        IF_OW_GT(8 , "ldr q0, [%[add_ptr], #128]\n\t"  )
        IF_OW_GT(8 , "add v10.8h, v10.8h, v0.8h\n\t"   )
        IF_OW_GT(9 , "ldr q1, [%[add_ptr], #144]\n\t"  )
        IF_OW_GT(9 , "add v11.8h, v11.8h, v1.8h\n\t"   )
        IF_OW_GT(10, "ldr q0, [%[add_ptr], #160]\n\t"  )
        IF_OW_GT(10, "add v12.8h, v12.8h, v0.8h\n\t"   )
        IF_OW_GT(11, "ldr q1, [%[add_ptr], #176]\n\t"  )
        IF_OW_GT(11, "add v13.8h, v13.8h, v1.8h\n\t"   )
        IF_OW_GT(12, "ldr q0, [%[add_ptr], #192]\n\t"  )
        IF_OW_GT(12, "add v14.8h, v14.8h, v0.8h\n\t"   )
        IF_OW_GT(13, "ldr q1, [%[add_ptr], #208]\n\t"  )
        IF_OW_GT(13, "add v15.8h, v15.8h, v1.8h\n\t"   )

        IF_OW_GT(0 , "ldr q0, [x12      ]\n\t"         )
        IF_OW_GT(0 , "add v16.8h, v16.8h, v0.8h\n\t"   )
        IF_OW_GT(1 , "ldr q1, [x12, #16 ]\n\t"         )
        IF_OW_GT(1 , "add v17.8h, v17.8h, v1.8h\n\t"   )
        IF_OW_GT(2 , "ldr q0, [x12, #32 ]\n\t"         )
        IF_OW_GT(2 , "add v18.8h, v18.8h, v0.8h\n\t"   )
        IF_OW_GT(3 , "ldr q1, [x12, #48 ]\n\t"         )
        IF_OW_GT(3 , "add v19.8h, v19.8h, v1.8h\n\t"   )
        IF_OW_GT(4 , "ldr q0, [x12, #64 ]\n\t"         )
        IF_OW_GT(4 , "add v20.8h, v20.8h, v0.8h\n\t"   )
        IF_OW_GT(5 , "ldr q1, [x12, #80 ]\n\t"         )
        IF_OW_GT(5 , "add v21.8h, v21.8h, v1.8h\n\t"   )
        IF_OW_GT(6 , "ldr q0, [x12, #96 ]\n\t"         )
        IF_OW_GT(6 , "add v22.8h, v22.8h, v0.8h\n\t"   )
        IF_OW_GT(7 , "ldr q1, [x12, #112]\n\t"         )
        IF_OW_GT(7 , "add v23.8h, v23.8h, v1.8h\n\t"   )
        IF_OW_GT(8 , "ldr q0, [x12, #128]\n\t"         )
        IF_OW_GT(8 , "add v24.8h, v24.8h, v0.8h\n\t"   )
        IF_OW_GT(9 , "ldr q1, [x12, #144]\n\t"         )
        IF_OW_GT(9 , "add v25.8h, v25.8h, v1.8h\n\t"   )
        IF_OW_GT(10, "ldr q0, [x12, #160]\n\t"         )
        IF_OW_GT(10, "add v26.8h, v26.8h, v0.8h\n\t"   )
        IF_OW_GT(11, "ldr q1, [x12, #176]\n\t"         )
        IF_OW_GT(11, "add v27.8h, v27.8h, v1.8h\n\t"   )
        IF_OW_GT(12, "ldr q0, [x12, #192]\n\t"         )
        IF_OW_GT(12, "add v28.8h, v28.8h, v0.8h\n\t"   )
        IF_OW_GT(13, "ldr q1, [x12, #208]\n\t"         )
        IF_OW_GT(13, "add v29.8h, v29.8h, v1.8h\n\t"   )

    "6:\n\t"
        "ands w13, %w[fuse_op], #2\n\t"
        "beq 7f\n\t"
        "fmov h0, 6.0e+0\n\t"
        "dup  v0.8h, v0.h[0]\n\t"
        
        IF_OW_GT(0 , "fmin v2.8h,  v2.8h,  v0.8h\n\t")
        IF_OW_GT(1 , "fmin v3.8h,  v3.8h,  v0.8h\n\t")
        IF_OW_GT(2 , "fmin v4.8h,  v4.8h,  v0.8h\n\t")
        IF_OW_GT(3 , "fmin v5.8h,  v5.8h,  v0.8h\n\t")
        IF_OW_GT(4 , "fmin v6.8h,  v6.8h,  v0.8h\n\t")
        IF_OW_GT(5 , "fmin v7.8h,  v7.8h,  v0.8h\n\t")
        IF_OW_GT(6 , "fmin v8.8h,  v8.8h,  v0.8h\n\t")
        IF_OW_GT(7 , "fmin v9.8h,  v9.8h,  v0.8h\n\t")
        IF_OW_GT(8 , "fmin v10.8h, v10.8h, v0.8h\n\t")
        IF_OW_GT(9 , "fmin v11.8h, v11.8h, v0.8h\n\t")
        IF_OW_GT(10, "fmin v12.8h, v12.8h, v0.8h\n\t")
        IF_OW_GT(11, "fmin v13.8h, v13.8h, v0.8h\n\t")
        IF_OW_GT(12, "fmin v14.8h, v14.8h, v0.8h\n\t")
        IF_OW_GT(13, "fmin v15.8h, v15.8h, v0.8h\n\t")
        IF_OW_GT(0 , "fmin v16.8h, v16.8h, v0.8h\n\t")
        IF_OW_GT(1 , "fmin v17.8h, v17.8h, v0.8h\n\t")
        IF_OW_GT(2 , "fmin v18.8h, v18.8h, v0.8h\n\t")
        IF_OW_GT(3 , "fmin v19.8h, v19.8h, v0.8h\n\t")
        IF_OW_GT(4 , "fmin v20.8h, v20.8h, v0.8h\n\t")
        IF_OW_GT(5 , "fmin v21.8h, v21.8h, v0.8h\n\t")
        IF_OW_GT(6 , "fmin v22.8h, v22.8h, v0.8h\n\t")
        IF_OW_GT(7 , "fmin v23.8h, v23.8h, v0.8h\n\t")
        IF_OW_GT(8 , "fmin v24.8h, v24.8h, v0.8h\n\t")
        IF_OW_GT(9 , "fmin v25.8h, v25.8h, v0.8h\n\t")
        IF_OW_GT(10, "fmin v26.8h, v26.8h, v0.8h\n\t")
        IF_OW_GT(11, "fmin v27.8h, v27.8h, v0.8h\n\t")
        IF_OW_GT(12, "fmin v28.8h, v28.8h, v0.8h\n\t")
        IF_OW_GT(13, "fmin v29.8h, v29.8h, v0.8h\n\t")
    
    "7:\n\t"
        "ands w13, %w[fuse_op], #1\n\t"
        "beq 8f\n\t"
        "eor v0.16b, v0.16b, v0.16b\n\t"
        
        IF_OW_GT(0 , "fmax v2.8h,  v2.8h,  v0.8h\n\t")
        IF_OW_GT(1 , "fmax v3.8h,  v3.8h,  v0.8h\n\t")
        IF_OW_GT(2 , "fmax v4.8h,  v4.8h,  v0.8h\n\t")
        IF_OW_GT(3 , "fmax v5.8h,  v5.8h,  v0.8h\n\t")
        IF_OW_GT(4 , "fmax v6.8h,  v6.8h,  v0.8h\n\t")
        IF_OW_GT(5 , "fmax v7.8h,  v7.8h,  v0.8h\n\t")
        IF_OW_GT(6 , "fmax v8.8h,  v8.8h,  v0.8h\n\t")
        IF_OW_GT(7 , "fmax v9.8h,  v9.8h,  v0.8h\n\t")
        IF_OW_GT(8 , "fmax v10.8h, v10.8h, v0.8h\n\t")
        IF_OW_GT(9 , "fmax v11.8h, v11.8h, v0.8h\n\t")
        IF_OW_GT(10, "fmax v12.8h, v12.8h, v0.8h\n\t")
        IF_OW_GT(11, "fmax v13.8h, v13.8h, v0.8h\n\t")
        IF_OW_GT(12, "fmax v14.8h, v14.8h, v0.8h\n\t")
        IF_OW_GT(13, "fmax v15.8h, v15.8h, v0.8h\n\t")
        IF_OW_GT(0 , "fmax v16.8h, v16.8h, v0.8h\n\t")
        IF_OW_GT(1 , "fmax v17.8h, v17.8h, v0.8h\n\t")
        IF_OW_GT(2 , "fmax v18.8h, v18.8h, v0.8h\n\t")
        IF_OW_GT(3 , "fmax v19.8h, v19.8h, v0.8h\n\t")
        IF_OW_GT(4 , "fmax v20.8h, v20.8h, v0.8h\n\t")
        IF_OW_GT(5 , "fmax v21.8h, v21.8h, v0.8h\n\t")
        IF_OW_GT(6 , "fmax v22.8h, v22.8h, v0.8h\n\t")
        IF_OW_GT(7 , "fmax v23.8h, v23.8h, v0.8h\n\t")
        IF_OW_GT(8 , "fmax v24.8h, v24.8h, v0.8h\n\t")
        IF_OW_GT(9 , "fmax v25.8h, v25.8h, v0.8h\n\t")
        IF_OW_GT(10, "fmax v26.8h, v26.8h, v0.8h\n\t")
        IF_OW_GT(11, "fmax v27.8h, v27.8h, v0.8h\n\t")
        IF_OW_GT(12, "fmax v28.8h, v28.8h, v0.8h\n\t")
        IF_OW_GT(13, "fmax v29.8h, v29.8h, v0.8h\n\t")

    "8:\n\t"
        "add x12, %[outPtr], %[out_bchw_stride_bytes]\n\t"
        IF_OW_GT(0,  "str q2,  [%[outPtr]      ]\n\t")
        IF_OW_GT(1,  "str q3,  [%[outPtr], #16 ]\n\t")
        IF_OW_GT(2,  "str q4,  [%[outPtr], #32 ]\n\t")
        IF_OW_GT(3,  "str q5,  [%[outPtr], #48 ]\n\t")
        IF_OW_GT(4,  "str q6,  [%[outPtr], #64 ]\n\t")
        IF_OW_GT(5,  "str q7,  [%[outPtr], #80 ]\n\t")
        IF_OW_GT(6,  "str q8,  [%[outPtr], #96 ]\n\t")
        IF_OW_GT(7,  "str q9,  [%[outPtr], #112]\n\t")
        IF_OW_GT(8,  "str q10, [%[outPtr], #128]\n\t")
        IF_OW_GT(9,  "str q11, [%[outPtr], #144]\n\t")
        IF_OW_GT(10, "str q12, [%[outPtr], #160]\n\t")
        IF_OW_GT(11, "str q13, [%[outPtr], #176]\n\t")
        IF_OW_GT(12, "str q14, [%[outPtr], #192]\n\t")
        IF_OW_GT(13, "str q15, [%[outPtr], #208]\n\t")

        IF_OW_GT(0,  "str q16, [x12      ]\n\t")
        IF_OW_GT(1,  "str q17, [x12, #16 ]\n\t")
        IF_OW_GT(2,  "str q18, [x12, #32 ]\n\t")
        IF_OW_GT(3,  "str q19, [x12, #48 ]\n\t")
        IF_OW_GT(4,  "str q20, [x12, #64 ]\n\t")
        IF_OW_GT(5,  "str q21, [x12, #80 ]\n\t")
        IF_OW_GT(6,  "str q22, [x12, #96 ]\n\t")
        IF_OW_GT(7,  "str q23, [x12, #112]\n\t")
        IF_OW_GT(8,  "str q24, [x12, #128]\n\t")
        IF_OW_GT(9,  "str q25, [x12, #144]\n\t")
        IF_OW_GT(10, "str q26, [x12, #160]\n\t")
        IF_OW_GT(11, "str q27, [x12, #176]\n\t")
        IF_OW_GT(12, "str q28, [x12, #192]\n\t")
        IF_OW_GT(13, "str q29, [x12, #208]\n\t")

    :
    :
      [flt_ptr]               "r" (filter_base),
      [biasPtr]               "r" (bias_base),
      [in_ptr]                "r" (input_base),
      [add_ptr]               "r" (sum_base),
      [outPtr]                "r" (output_base),
      [flt_h_valid]           "r" (fltH_valid),
      [flt_w]                 "r" (fltW),
      [out_bchw_stride_bytes] "r" (outBCHW_stride * 2),
      [flt_ic_stride_bytes]   "r" (filter_ic_stride * 2),
      [hw_in_bytes]           "r" (inH * inW * 2),
      [h_dltn_w_in_bytes]     "r" (dltnH * inW * 2),
      [w_dltn_bytes]          "r" (dltnW * 2),
      [w_strd_bytes]          "r" (strdW * 2),
      [ic] "r" (inC),
      [fuse_op] "r" (fuse_type)
    :
      "cc",
      "memory"
 #ifndef __INTELLISENSE__
          ,
          "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",
          "v10" , "v11" , "v12" , "v13" , "v14" , "v15" , "v16" , "v17" ,
          "v18" , "v19" , "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v0", "v1", "v2", "v3",
          "v4", "v5", "v6", "v7", "v8", "v9", "v30", "v31"
#endif     
    );
}

template<>
void ppl_kernel_arm_server_conv2d_fp16_conv_direct_ndarray_kernel<8, OUT_TILE_W()>(
    const __fp16 *input_base,
    const __fp16 *filter_base,
    const __fp16 *bias_base,
    __fp16 *output_base,
    __fp16 *sum_base,
    const int64_t inH,
    const int64_t inW,
    const int64_t inC,
    const int64_t fltH_valid,
    const int64_t fltW,
    const int64_t strdW,
    const int64_t dltnH,
    const int64_t dltnW,
    const int64_t filter_ic_stride,
    const int64_t outBCHW_stride,
    const uint32_t fuse_type)
{
    float16x8_t v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 ; (void)v0 ; (void)v1 ; (void)v2 ; (void)v3 ; (void)v4 ; (void)v5 ; (void)v6 ; (void)v7 ;
    float16x8_t v8 , v9 , v10, v11, v12, v13, v14, v15; (void)v8 ; (void)v9 ; (void)v10; (void)v11; (void)v12; (void)v13; (void)v14; (void)v15;
    float16x8_t v16, v17, v18, v19, v20, v21, v22, v23; (void)v16; (void)v17; (void)v18; (void)v19; (void)v20; (void)v21; (void)v22; (void)v23;
    float16x8_t v24, v25, v26, v27, v28, v29, v30, v31; (void)v24; (void)v25; (void)v26; (void)v27; (void)v28; (void)v29; (void)v30; (void)v31;

        if (nullptr == bias_base) {
            __fp16 *output_oc0_base = output_base;
#if OUT_TILE_W() > 0
            v0  = vld1q_f16(output_oc0_base + 0  * CBLK());
#endif
#if OUT_TILE_W() > 1
            v1  = vld1q_f16(output_oc0_base + 1  * CBLK());
#endif
#if OUT_TILE_W() > 2
            v2  = vld1q_f16(output_oc0_base + 2  * CBLK());
#endif
#if OUT_TILE_W() > 3
            v3  = vld1q_f16(output_oc0_base + 3  * CBLK());
#endif
#if OUT_TILE_W() > 4
            v4  = vld1q_f16(output_oc0_base + 4  * CBLK());
#endif
#if OUT_TILE_W() > 5
            v5  = vld1q_f16(output_oc0_base + 5  * CBLK());
#endif
#if OUT_TILE_W() > 6
            v6  = vld1q_f16(output_oc0_base + 6  * CBLK());
#endif
#if OUT_TILE_W() > 7
            v7  = vld1q_f16(output_oc0_base + 7  * CBLK());
#endif
#if OUT_TILE_W() > 8
            v8  = vld1q_f16(output_oc0_base + 8  * CBLK());
#endif
#if OUT_TILE_W() > 9
            v9  = vld1q_f16(output_oc0_base + 9  * CBLK());
#endif
#if OUT_TILE_W() > 10
            v10 = vld1q_f16(output_oc0_base + 10 * CBLK());
#endif
#if OUT_TILE_W() > 11
            v11 = vld1q_f16(output_oc0_base + 11 * CBLK());
#endif
#if OUT_TILE_W() > 12
            v12 = vld1q_f16(output_oc0_base + 12 * CBLK());
#endif
#if OUT_TILE_W() > 13
            v13 = vld1q_f16(output_oc0_base + 13 * CBLK());
#endif
        }  // load last output
        else {
#if OUT_TILE_W() > 0
            v0  = vld1q_f16(bias_base);
#endif
#if OUT_TILE_W() > 1
            v1  = v0;
#endif
#if OUT_TILE_W() > 2
            v2  = v0;
#endif
#if OUT_TILE_W() > 3
            v3  = v0;
#endif
#if OUT_TILE_W() > 4
            v4  = v0;
#endif
#if OUT_TILE_W() > 5
            v5  = v0;
#endif
#if OUT_TILE_W() > 6
            v6  = v0;
#endif
#if OUT_TILE_W() > 7
            v7  = v0;
#endif
#if OUT_TILE_W() > 8
            v8  = v0;
#endif
#if OUT_TILE_W() > 9
            v9  = v0;
#endif
#if OUT_TILE_W() > 10
            v10 = v0;
#endif
#if OUT_TILE_W() > 11
            v11 = v0;
#endif
#if OUT_TILE_W() > 12
            v12 = v0;
#endif
#if OUT_TILE_W() > 13
            v13 = v0;
#endif
        }  // load bias as output

        const __fp16 *input_ic_base = input_base;
        int64_t ic = inC;
        const __fp16 *filter_ic_base = filter_base;
        do {
            const __fp16 *input_kh_base = input_ic_base;
            const __fp16 *filter_ptr = filter_ic_base;
            for (int64_t kh = 0; kh < fltH_valid; kh++) {
                const __fp16 *input_kw_base = input_kh_base;
                for (int64_t kw = 0; kw < fltW; kw++) {
                    const __fp16 *input_ptr = input_kw_base;
                    v28 = vld1q_f16(filter_ptr);
#if OUT_TILE_W() > 0
                    v30 = vld1q_dup_f16(input_ptr);
                    v0  = vfmaq_f16(v0 , v28, v30);
#endif
#if OUT_TILE_W() > 1
                    v31 = vld1q_dup_f16(input_ptr + strdW);
                    v1  = vfmaq_f16(v1 , v28, v31);
#endif
#if OUT_TILE_W() > 2
                    v30 = vld1q_dup_f16(input_ptr + 2 * strdW);
                    v2  = vfmaq_f16(v2 , v28, v30);
#endif
#if OUT_TILE_W() > 3
                    v31 = vld1q_dup_f16(input_ptr + 3 * strdW);
                    v3  = vfmaq_f16(v3 , v28, v31);
#endif
#if OUT_TILE_W() > 4
                    v30 = vld1q_dup_f16(input_ptr + 4 * strdW);
                    v4  = vfmaq_f16(v4 , v28, v30);
#endif
#if OUT_TILE_W() > 5
                    v31 = vld1q_dup_f16(input_ptr + 5 * strdW);
                    v5  = vfmaq_f16(v5 , v28, v31);
#endif
#if OUT_TILE_W() > 6
                    v30 = vld1q_dup_f16(input_ptr + 6 * strdW);
                    v6  = vfmaq_f16(v6 , v28, v30);
#endif
#if OUT_TILE_W() > 7
                    v31 = vld1q_dup_f16(input_ptr + 7 * strdW);
                    v7  = vfmaq_f16(v7 , v28, v31);
#endif
#if OUT_TILE_W() > 8
                    v30 = vld1q_dup_f16(input_ptr + 8 * strdW);
                    v8  = vfmaq_f16(v8 , v28, v30);
#endif
#if OUT_TILE_W() > 9
                    v31 = vld1q_dup_f16(input_ptr + 9 * strdW);
                    v9  = vfmaq_f16(v9 , v28, v31);
#endif
#if OUT_TILE_W() > 10
                    v30 = vld1q_dup_f16(input_ptr + 10 * strdW);
                    v10 = vfmaq_f16(v10, v28, v30);
#endif
#if OUT_TILE_W() > 11
                    v31 = vld1q_dup_f16(input_ptr + 11 * strdW);
                    v11 = vfmaq_f16(v11, v28, v31);
#endif
#if OUT_TILE_W() > 12
                    v30 = vld1q_dup_f16(input_ptr + 12 * strdW);
                    v12 = vfmaq_f16(v12, v28, v30);
#endif
#if OUT_TILE_W() > 13
                    v31 = vld1q_dup_f16(input_ptr + 13 * strdW);
                    v13 = vfmaq_f16(v13, v28, v31);
#endif
                    input_kw_base += dltnW;
                    filter_ptr += CBLK() * 2;
                }
                input_kh_base += dltnH * inW;
            }
            input_ic_base += (inH * inW);
            filter_ic_base += filter_ic_stride;
            ic -= 1;
        }
        while (ic > 0);

        if (fuse_type & conv_fuse_flag::SUM) { // sum
#if OUT_TILE_W() > 0
            v0  = vaddq_f16(v0 , vld1q_f16(sum_base + 0  * CBLK()));
#endif
#if OUT_TILE_W() > 1
            v1  = vaddq_f16(v1 , vld1q_f16(sum_base + 1  * CBLK()));
#endif
#if OUT_TILE_W() > 2
            v2  = vaddq_f16(v2 , vld1q_f16(sum_base + 2  * CBLK()));
#endif
#if OUT_TILE_W() > 3
            v3  = vaddq_f16(v3 , vld1q_f16(sum_base + 3  * CBLK()));
#endif
#if OUT_TILE_W() > 4
            v4  = vaddq_f16(v4 , vld1q_f16(sum_base + 4  * CBLK()));
#endif
#if OUT_TILE_W() > 5
            v5  = vaddq_f16(v5 , vld1q_f16(sum_base + 5  * CBLK()));
#endif
#if OUT_TILE_W() > 6
            v6  = vaddq_f16(v6 , vld1q_f16(sum_base + 6  * CBLK()));
#endif
#if OUT_TILE_W() > 7
            v7  = vaddq_f16(v7 , vld1q_f16(sum_base + 7  * CBLK()));
#endif
#if OUT_TILE_W() > 8
            v8  = vaddq_f16(v8 , vld1q_f16(sum_base + 8  * CBLK()));
#endif
#if OUT_TILE_W() > 9
            v9  = vaddq_f16(v9 , vld1q_f16(sum_base + 9  * CBLK()));
#endif
#if OUT_TILE_W() > 10
            v10 = vaddq_f16(v10, vld1q_f16(sum_base + 10 * CBLK()));
#endif
#if OUT_TILE_W() > 11
            v11 = vaddq_f16(v11, vld1q_f16(sum_base + 11 * CBLK()));
#endif
#if OUT_TILE_W() > 12
            v12 = vaddq_f16(v12, vld1q_f16(sum_base + 12 * CBLK()));
#endif
#if OUT_TILE_W() > 13
            v13 = vaddq_f16(v13, vld1q_f16(sum_base + 13 * CBLK()));
#endif            
        }

        if (fuse_type & conv_fuse_flag::RELU) { // relu
            v31 = vdupq_n_f16(0.0f);
#if OUT_TILE_W() > 0
            v0  = vmaxq_f16(v0 , v31);
#endif
#if OUT_TILE_W() > 1
            v1  = vmaxq_f16(v1 , v31);
#endif
#if OUT_TILE_W() > 2
            v2  = vmaxq_f16(v2 , v31);
#endif
#if OUT_TILE_W() > 3
            v3  = vmaxq_f16(v3 , v31);
#endif
#if OUT_TILE_W() > 4
            v4  = vmaxq_f16(v4 , v31);
#endif
#if OUT_TILE_W() > 5
            v5  = vmaxq_f16(v5 , v31);
#endif
#if OUT_TILE_W() > 6
            v6  = vmaxq_f16(v6 , v31);
#endif
#if OUT_TILE_W() > 7
            v7  = vmaxq_f16(v7 , v31);
#endif
#if OUT_TILE_W() > 8
            v8  = vmaxq_f16(v8 , v31);
#endif
#if OUT_TILE_W() > 9
            v9  = vmaxq_f16(v9 , v31);
#endif
#if OUT_TILE_W() > 10
            v10 = vmaxq_f16(v10, v31);
#endif
#if OUT_TILE_W() > 11
            v11 = vmaxq_f16(v11, v31);
#endif
#if OUT_TILE_W() > 12
            v12 = vmaxq_f16(v12, v31);
#endif
#if OUT_TILE_W() > 13
            v13 = vmaxq_f16(v13, v31);
#endif
        }
        if (fuse_type & conv_fuse_flag::RELU6) { // relu6
            v31 = vdupq_n_f16(6.0f);
#if OUT_TILE_W() > 0
            v0  = vminq_f16(v0 , v31);
#endif
#if OUT_TILE_W() > 1
            v1  = vminq_f16(v1 , v31);
#endif
#if OUT_TILE_W() > 2
            v2  = vminq_f16(v2 , v31);
#endif
#if OUT_TILE_W() > 3
            v3  = vminq_f16(v3 , v31);
#endif
#if OUT_TILE_W() > 4
            v4  = vminq_f16(v4 , v31);
#endif
#if OUT_TILE_W() > 5
            v5  = vminq_f16(v5 , v31);
#endif
#if OUT_TILE_W() > 6
            v6  = vminq_f16(v6 , v31);
#endif
#if OUT_TILE_W() > 7
            v7  = vminq_f16(v7 , v31);
#endif
#if OUT_TILE_W() > 8
            v8  = vminq_f16(v8 , v31);
#endif
#if OUT_TILE_W() > 9
            v9  = vminq_f16(v9 , v31);
#endif
#if OUT_TILE_W() > 10
            v10 = vminq_f16(v10, v31);
#endif
#if OUT_TILE_W() > 11
            v11 = vminq_f16(v11, v31);
#endif
#if OUT_TILE_W() > 12
            v12 = vminq_f16(v12, v31);
#endif
#if OUT_TILE_W() > 13
            v13 = vminq_f16(v13, v31);
#endif
        }

        __fp16 *output_oc0_base = output_base;
#if OUT_TILE_W() > 0
        vst1q_f16(output_oc0_base + 0  * CBLK(), v0 );
#endif
#if OUT_TILE_W() > 1
        vst1q_f16(output_oc0_base + 1  * CBLK(), v1 );
#endif
#if OUT_TILE_W() > 2
        vst1q_f16(output_oc0_base + 2  * CBLK(), v2 );
#endif
#if OUT_TILE_W() > 3
        vst1q_f16(output_oc0_base + 3  * CBLK(), v3 );
#endif
#if OUT_TILE_W() > 4
        vst1q_f16(output_oc0_base + 4  * CBLK(), v4 );
#endif
#if OUT_TILE_W() > 5
        vst1q_f16(output_oc0_base + 5  * CBLK(), v5 );
#endif
#if OUT_TILE_W() > 6
        vst1q_f16(output_oc0_base + 6  * CBLK(), v6 );
#endif
#if OUT_TILE_W() > 7
        vst1q_f16(output_oc0_base + 7  * CBLK(), v7 );
#endif
#if OUT_TILE_W() > 8
        vst1q_f16(output_oc0_base + 8  * CBLK(), v8 );
#endif
#if OUT_TILE_W() > 9
        vst1q_f16(output_oc0_base + 9  * CBLK(), v9 );
#endif
#if OUT_TILE_W() > 10
        vst1q_f16(output_oc0_base + 10 * CBLK(), v10);
#endif
#if OUT_TILE_W() > 11
        vst1q_f16(output_oc0_base + 11 * CBLK(), v11);
#endif
#if OUT_TILE_W() > 12
        vst1q_f16(output_oc0_base + 12 * CBLK(), v12);
#endif
#if OUT_TILE_W() > 13
        vst1q_f16(output_oc0_base + 13 * CBLK(), v13);
#endif
}