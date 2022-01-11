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

template<>
void ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<8, OUT_TILE_W()>(
    const float *input_base,
    const float *filter_base,
    const float *bias_base,
    float *output_base,
    float *sum_base,
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
    float32x4_t v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 ; (void)v0 ; (void)v1 ; (void)v2 ; (void)v3 ; (void)v4 ; (void)v5 ; (void)v6 ; (void)v7 ;
    float32x4_t v8 , v9 , v10, v11, v12, v13, v14, v15; (void)v8 ; (void)v9 ; (void)v10; (void)v11; (void)v12; (void)v13; (void)v14; (void)v15;
    float32x4_t v16, v17, v18, v19, v20, v21, v22, v23; (void)v16; (void)v17; (void)v18; (void)v19; (void)v20; (void)v21; (void)v22; (void)v23;
    float32x4_t v24, v25, v26, v27, v28, v29, v30, v31; (void)v24; (void)v25; (void)v26; (void)v27; (void)v28; (void)v29; (void)v30; (void)v31;

    
        if (nullptr == bias_base) {
            float *output_oc0_base = output_base;
#if OUT_TILE_W() > 0
            v0  = vld1q_f32(output_oc0_base + 0  * CBLK());
#endif
#if OUT_TILE_W() > 1
            v1  = vld1q_f32(output_oc0_base + 1  * CBLK());
#endif
#if OUT_TILE_W() > 2
            v2  = vld1q_f32(output_oc0_base + 2  * CBLK());
#endif
#if OUT_TILE_W() > 3
            v3  = vld1q_f32(output_oc0_base + 3  * CBLK());
#endif
#if OUT_TILE_W() > 4
            v4  = vld1q_f32(output_oc0_base + 4  * CBLK());
#endif
#if OUT_TILE_W() > 5
            v5  = vld1q_f32(output_oc0_base + 5  * CBLK());
#endif
#if OUT_TILE_W() > 6
            v6  = vld1q_f32(output_oc0_base + 6  * CBLK());
#endif
#if OUT_TILE_W() > 7
            v7  = vld1q_f32(output_oc0_base + 7  * CBLK());
#endif
#if OUT_TILE_W() > 8
            v8  = vld1q_f32(output_oc0_base + 8  * CBLK());
#endif
#if OUT_TILE_W() > 9
            v9  = vld1q_f32(output_oc0_base + 9  * CBLK());
#endif
#if OUT_TILE_W() > 10
            v10 = vld1q_f32(output_oc0_base + 10 * CBLK());
#endif
#if OUT_TILE_W() > 11
            v11 = vld1q_f32(output_oc0_base + 11 * CBLK());
#endif
#if OUT_TILE_W() > 12
            v12 = vld1q_f32(output_oc0_base + 12 * CBLK());
#endif
#if OUT_TILE_W() > 13
            v13 = vld1q_f32(output_oc0_base + 13 * CBLK());
#endif
            float *output_oc1_base = output_oc0_base + outBCHW_stride;
#if OUT_TILE_W() > 0
            v14 = vld1q_f32(output_oc1_base + 0  * CBLK());
#endif
#if OUT_TILE_W() > 1
            v15 = vld1q_f32(output_oc1_base + 1  * CBLK());
#endif
#if OUT_TILE_W() > 2
            v16 = vld1q_f32(output_oc1_base + 2  * CBLK());
#endif
#if OUT_TILE_W() > 3
            v17 = vld1q_f32(output_oc1_base + 3  * CBLK());
#endif
#if OUT_TILE_W() > 4
            v18 = vld1q_f32(output_oc1_base + 4  * CBLK());
#endif
#if OUT_TILE_W() > 5
            v19 = vld1q_f32(output_oc1_base + 5  * CBLK());
#endif
#if OUT_TILE_W() > 6
            v20 = vld1q_f32(output_oc1_base + 6  * CBLK());
#endif
#if OUT_TILE_W() > 7
            v21 = vld1q_f32(output_oc1_base + 7  * CBLK());
#endif
#if OUT_TILE_W() > 8
            v22 = vld1q_f32(output_oc1_base + 8  * CBLK());
#endif
#if OUT_TILE_W() > 9
            v23 = vld1q_f32(output_oc1_base + 9  * CBLK());
#endif
#if OUT_TILE_W() > 10
            v24 = vld1q_f32(output_oc1_base + 10 * CBLK());
#endif
#if OUT_TILE_W() > 11
            v25 = vld1q_f32(output_oc1_base + 11 * CBLK());
#endif
#if OUT_TILE_W() > 12
            v26 = vld1q_f32(output_oc1_base + 12 * CBLK());
#endif
#if OUT_TILE_W() > 13
            v27 = vld1q_f32(output_oc1_base + 13 * CBLK());
#endif
        }  // load last output
        else {
#if OUT_TILE_W() > 0
            v0  = vld1q_f32(bias_base);
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

#if OUT_TILE_W() > 0
            v14 = vld1q_f32(bias_base + CBLK());
#endif
#if OUT_TILE_W() > 1
            v15 = v14;
#endif
#if OUT_TILE_W() > 2
            v16 = v14;
#endif
#if OUT_TILE_W() > 3
            v17 = v14;
#endif
#if OUT_TILE_W() > 4
            v18 = v14;
#endif
#if OUT_TILE_W() > 5
            v19 = v14;
#endif
#if OUT_TILE_W() > 6
            v20 = v14;
#endif
#if OUT_TILE_W() > 7
            v21 = v14;
#endif
#if OUT_TILE_W() > 8
            v22 = v14;
#endif
#if OUT_TILE_W() > 9
            v23 = v14;
#endif
#if OUT_TILE_W() > 10
            v24 = v14;
#endif
#if OUT_TILE_W() > 11
            v25 = v14;
#endif
#if OUT_TILE_W() > 12
            v26 = v14;
#endif
#if OUT_TILE_W() > 13
            v27 = v14;
#endif
        }  // load bias as output

        const float *input_ic_base = input_base;
        int64_t ic = inC;
        const float *filter_ic_base = filter_base;
        do {
            const float *input_kh_base = input_ic_base;
            const float *filter_ptr = filter_ic_base;
            for (int64_t kh = 0; kh < fltH_valid; kh++) {
                const float *input_kw_base = input_kh_base;
                for (int64_t kw = 0; kw < fltW; kw++) {
                    const float *input_ptr = input_kw_base;
                    v28 = vld1q_f32(filter_ptr);
                    v29 = vld1q_f32(filter_ptr + CBLK());
#if OUT_TILE_W() > 0
                    v30 = vld1q_dup_f32(input_ptr);
                    v0  = vfmaq_f32(v0 , v28, v30);
                    v14 = vfmaq_f32(v14, v29, v30);
#endif
#if OUT_TILE_W() > 1
                    v31 = vld1q_dup_f32(input_ptr + strdW);
                    v1  = vfmaq_f32(v1 , v28, v31);
                    v15 = vfmaq_f32(v15, v29, v31);
#endif
#if OUT_TILE_W() > 2
                    v30 = vld1q_dup_f32(input_ptr + 2 * strdW);
                    v2  = vfmaq_f32(v2 , v28, v30);
                    v16 = vfmaq_f32(v16, v29, v30);
#endif
#if OUT_TILE_W() > 3
                    v31 = vld1q_dup_f32(input_ptr + 3 * strdW);
                    v3  = vfmaq_f32(v3 , v28, v31);
                    v17 = vfmaq_f32(v17, v29, v31);
#endif
#if OUT_TILE_W() > 4
                    v30 = vld1q_dup_f32(input_ptr + 4 * strdW);
                    v4  = vfmaq_f32(v4 , v28, v30);
                    v18 = vfmaq_f32(v18, v29, v30);
#endif
#if OUT_TILE_W() > 5
                    v31 = vld1q_dup_f32(input_ptr + 5 * strdW);
                    v5  = vfmaq_f32(v5 , v28, v31);
                    v19 = vfmaq_f32(v19, v29, v31);
#endif
#if OUT_TILE_W() > 6
                    v30 = vld1q_dup_f32(input_ptr + 6 * strdW);
                    v6  = vfmaq_f32(v6 , v28, v30);
                    v20 = vfmaq_f32(v20, v29, v30);
#endif
#if OUT_TILE_W() > 7
                    v31 = vld1q_dup_f32(input_ptr + 7 * strdW);
                    v7  = vfmaq_f32(v7 , v28, v31);
                    v21 = vfmaq_f32(v21, v29, v31);
#endif
#if OUT_TILE_W() > 8
                    v30 = vld1q_dup_f32(input_ptr + 8 * strdW);
                    v8  = vfmaq_f32(v8 , v28, v30);
                    v22 = vfmaq_f32(v22, v29, v30);
#endif
#if OUT_TILE_W() > 9
                    v31 = vld1q_dup_f32(input_ptr + 9 * strdW);
                    v9  = vfmaq_f32(v9 , v28, v31);
                    v23 = vfmaq_f32(v23, v29, v31);
#endif
#if OUT_TILE_W() > 10
                    v30 = vld1q_dup_f32(input_ptr + 10 * strdW);
                    v10 = vfmaq_f32(v10, v28, v30);
                    v24 = vfmaq_f32(v24, v29, v30);
#endif
#if OUT_TILE_W() > 11
                    v31 = vld1q_dup_f32(input_ptr + 11 * strdW);
                    v11 = vfmaq_f32(v11, v28, v31);
                    v25 = vfmaq_f32(v25, v29, v31);
#endif
#if OUT_TILE_W() > 12
                    v30 = vld1q_dup_f32(input_ptr + 12 * strdW);
                    v12 = vfmaq_f32(v12, v28, v30);
                    v26 = vfmaq_f32(v26, v29, v30);
#endif
#if OUT_TILE_W() > 13
                    v31 = vld1q_dup_f32(input_ptr + 13 * strdW);
                    v13 = vfmaq_f32(v13, v28, v31);
                    v27 = vfmaq_f32(v27, v29, v31);
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
            v0  = vaddq_f32(v0 , vld1q_f32(sum_base + 0  * CBLK()));
#endif
#if OUT_TILE_W() > 1
            v1  = vaddq_f32(v1 , vld1q_f32(sum_base + 1  * CBLK()));
#endif
#if OUT_TILE_W() > 2
            v2  = vaddq_f32(v2 , vld1q_f32(sum_base + 2  * CBLK()));
#endif
#if OUT_TILE_W() > 3
            v3  = vaddq_f32(v3 , vld1q_f32(sum_base + 3  * CBLK()));
#endif
#if OUT_TILE_W() > 4
            v4  = vaddq_f32(v4 , vld1q_f32(sum_base + 4  * CBLK()));
#endif
#if OUT_TILE_W() > 5
            v5  = vaddq_f32(v5 , vld1q_f32(sum_base + 5  * CBLK()));
#endif
#if OUT_TILE_W() > 6
            v6  = vaddq_f32(v6 , vld1q_f32(sum_base + 6  * CBLK()));
#endif
#if OUT_TILE_W() > 7
            v7  = vaddq_f32(v7 , vld1q_f32(sum_base + 7  * CBLK()));
#endif
#if OUT_TILE_W() > 8
            v8  = vaddq_f32(v8 , vld1q_f32(sum_base + 8  * CBLK()));
#endif
#if OUT_TILE_W() > 9
            v9  = vaddq_f32(v9 , vld1q_f32(sum_base + 9  * CBLK()));
#endif
#if OUT_TILE_W() > 10
            v10 = vaddq_f32(v10, vld1q_f32(sum_base + 10 * CBLK()));
#endif
#if OUT_TILE_W() > 11
            v11 = vaddq_f32(v11, vld1q_f32(sum_base + 11 * CBLK()));
#endif
#if OUT_TILE_W() > 12
            v12 = vaddq_f32(v12, vld1q_f32(sum_base + 12 * CBLK()));
#endif
#if OUT_TILE_W() > 13
            v13 = vaddq_f32(v13, vld1q_f32(sum_base + 13 * CBLK()));
#endif
            const float *sum_ptr = sum_base + outBCHW_stride;
#if OUT_TILE_W() > 0
            v14 = vaddq_f32(v14, vld1q_f32(sum_ptr  + 0  * CBLK()));
#endif
#if OUT_TILE_W() > 1
            v15 = vaddq_f32(v15, vld1q_f32(sum_ptr  + 1  * CBLK()));
#endif
#if OUT_TILE_W() > 2
            v16 = vaddq_f32(v16, vld1q_f32(sum_ptr  + 2  * CBLK()));
#endif
#if OUT_TILE_W() > 3
            v17 = vaddq_f32(v17, vld1q_f32(sum_ptr  + 3  * CBLK()));
#endif
#if OUT_TILE_W() > 4
            v18 = vaddq_f32(v18, vld1q_f32(sum_ptr  + 4  * CBLK()));
#endif
#if OUT_TILE_W() > 5
            v19 = vaddq_f32(v19, vld1q_f32(sum_ptr  + 5  * CBLK()));
#endif
#if OUT_TILE_W() > 6
            v20 = vaddq_f32(v20, vld1q_f32(sum_ptr  + 6  * CBLK()));
#endif
#if OUT_TILE_W() > 7
            v21 = vaddq_f32(v21, vld1q_f32(sum_ptr  + 7  * CBLK()));
#endif
#if OUT_TILE_W() > 8
            v22 = vaddq_f32(v22, vld1q_f32(sum_ptr  + 8  * CBLK()));
#endif
#if OUT_TILE_W() > 9
            v23 = vaddq_f32(v23, vld1q_f32(sum_ptr  + 9  * CBLK()));
#endif
#if OUT_TILE_W() > 10
            v24 = vaddq_f32(v24, vld1q_f32(sum_ptr  + 10 * CBLK()));
#endif
#if OUT_TILE_W() > 11
            v25 = vaddq_f32(v25, vld1q_f32(sum_ptr  + 11 * CBLK()));
#endif
#if OUT_TILE_W() > 12
            v26 = vaddq_f32(v26, vld1q_f32(sum_ptr  + 12 * CBLK()));
#endif
#if OUT_TILE_W() > 13
            v27 = vaddq_f32(v27, vld1q_f32(sum_ptr  + 13 * CBLK()));
#endif
        }
        if (fuse_type & conv_fuse_flag::RELU) { // relu
            v31 = vdupq_n_f32(0.0f);
#if OUT_TILE_W() > 0
            v0  = vmaxq_f32(v0 , v31);
            v14 = vmaxq_f32(v14, v31);
#endif
#if OUT_TILE_W() > 1
            v1  = vmaxq_f32(v1 , v31);
            v15 = vmaxq_f32(v15, v31);
#endif
#if OUT_TILE_W() > 2
            v2  = vmaxq_f32(v2 , v31);
            v16 = vmaxq_f32(v16, v31);
#endif
#if OUT_TILE_W() > 3
            v3  = vmaxq_f32(v3 , v31);
            v17 = vmaxq_f32(v17, v31);
#endif
#if OUT_TILE_W() > 4
            v4  = vmaxq_f32(v4 , v31);
            v18 = vmaxq_f32(v18, v31);
#endif
#if OUT_TILE_W() > 5
            v5  = vmaxq_f32(v5 , v31);
            v19 = vmaxq_f32(v19, v31);
#endif
#if OUT_TILE_W() > 6
            v6  = vmaxq_f32(v6 , v31);
            v20 = vmaxq_f32(v20, v31);
#endif
#if OUT_TILE_W() > 7
            v7  = vmaxq_f32(v7 , v31);
            v21 = vmaxq_f32(v21, v31);
#endif
#if OUT_TILE_W() > 8
            v8  = vmaxq_f32(v8 , v31);
            v22 = vmaxq_f32(v22, v31);
#endif
#if OUT_TILE_W() > 9
            v9  = vmaxq_f32(v9 , v31);
            v23 = vmaxq_f32(v23, v31);
#endif
#if OUT_TILE_W() > 10
            v10 = vmaxq_f32(v10, v31);
            v24 = vmaxq_f32(v24, v31);
#endif
#if OUT_TILE_W() > 11
            v11 = vmaxq_f32(v11, v31);
            v25 = vmaxq_f32(v25, v31);
#endif
#if OUT_TILE_W() > 12
            v12 = vmaxq_f32(v12, v31);
            v26 = vmaxq_f32(v26, v31);
#endif
#if OUT_TILE_W() > 13
            v13 = vmaxq_f32(v13, v31);
            v27 = vmaxq_f32(v27, v31);
#endif
        }
        if (fuse_type & conv_fuse_flag::RELU6) { // relu6
            v31 = vdupq_n_f32(6.0f);
#if OUT_TILE_W() > 0
            v0  = vminq_f32(v0 , v31);
            v14 = vminq_f32(v14, v31);
#endif
#if OUT_TILE_W() > 1
            v1  = vminq_f32(v1 , v31);
            v15 = vminq_f32(v15, v31);
#endif
#if OUT_TILE_W() > 2
            v2  = vminq_f32(v2 , v31);
            v16 = vminq_f32(v16, v31);
#endif
#if OUT_TILE_W() > 3
            v3  = vminq_f32(v3 , v31);
            v17 = vminq_f32(v17, v31);
#endif
#if OUT_TILE_W() > 4
            v4  = vminq_f32(v4 , v31);
            v18 = vminq_f32(v18, v31);
#endif
#if OUT_TILE_W() > 5
            v5  = vminq_f32(v5 , v31);
            v19 = vminq_f32(v19, v31);
#endif
#if OUT_TILE_W() > 6
            v6  = vminq_f32(v6 , v31);
            v20 = vminq_f32(v20, v31);
#endif
#if OUT_TILE_W() > 7
            v7  = vminq_f32(v7 , v31);
            v21 = vminq_f32(v21, v31);
#endif
#if OUT_TILE_W() > 8
            v8  = vminq_f32(v8 , v31);
            v22 = vminq_f32(v22, v31);
#endif
#if OUT_TILE_W() > 9
            v9  = vminq_f32(v9 , v31);
            v23 = vminq_f32(v23, v31);
#endif
#if OUT_TILE_W() > 10
            v10 = vminq_f32(v10, v31);
            v24 = vminq_f32(v24, v31);
#endif
#if OUT_TILE_W() > 11
            v11 = vminq_f32(v11, v31);
            v25 = vminq_f32(v25, v31);
#endif
#if OUT_TILE_W() > 12
            v12 = vminq_f32(v12, v31);
            v26 = vminq_f32(v26, v31);
#endif
#if OUT_TILE_W() > 13
            v13 = vminq_f32(v13, v31);
            v27 = vminq_f32(v27, v31);
#endif
        }

        float *output_oc0_base = output_base;
#if OUT_TILE_W() > 0
        vst1q_f32(output_oc0_base + 0  * CBLK(), v0 );
#endif
#if OUT_TILE_W() > 1
        vst1q_f32(output_oc0_base + 1  * CBLK(), v1 );
#endif
#if OUT_TILE_W() > 2
        vst1q_f32(output_oc0_base + 2  * CBLK(), v2 );
#endif
#if OUT_TILE_W() > 3
        vst1q_f32(output_oc0_base + 3  * CBLK(), v3 );
#endif
#if OUT_TILE_W() > 4
        vst1q_f32(output_oc0_base + 4  * CBLK(), v4 );
#endif
#if OUT_TILE_W() > 5
        vst1q_f32(output_oc0_base + 5  * CBLK(), v5 );
#endif
#if OUT_TILE_W() > 6
        vst1q_f32(output_oc0_base + 6  * CBLK(), v6 );
#endif
#if OUT_TILE_W() > 7
        vst1q_f32(output_oc0_base + 7  * CBLK(), v7 );
#endif
#if OUT_TILE_W() > 8
        vst1q_f32(output_oc0_base + 8  * CBLK(), v8 );
#endif
#if OUT_TILE_W() > 9
        vst1q_f32(output_oc0_base + 9  * CBLK(), v9 );
#endif
#if OUT_TILE_W() > 10
        vst1q_f32(output_oc0_base + 10 * CBLK(), v10);
#endif
#if OUT_TILE_W() > 11
        vst1q_f32(output_oc0_base + 11 * CBLK(), v11);
#endif
#if OUT_TILE_W() > 12
        vst1q_f32(output_oc0_base + 12 * CBLK(), v12);
#endif
#if OUT_TILE_W() > 13
        vst1q_f32(output_oc0_base + 13 * CBLK(), v13);
#endif
        float *output_oc1_base = output_oc0_base + outBCHW_stride;
#if OUT_TILE_W() > 0
        vst1q_f32(output_oc1_base + 0  * CBLK(), v14);
#endif
#if OUT_TILE_W() > 1
        vst1q_f32(output_oc1_base + 1  * CBLK(), v15);
#endif
#if OUT_TILE_W() > 2
        vst1q_f32(output_oc1_base + 2  * CBLK(), v16);
#endif
#if OUT_TILE_W() > 3
        vst1q_f32(output_oc1_base + 3  * CBLK(), v17);
#endif
#if OUT_TILE_W() > 4
        vst1q_f32(output_oc1_base + 4  * CBLK(), v18);
#endif
#if OUT_TILE_W() > 5
        vst1q_f32(output_oc1_base + 5  * CBLK(), v19);
#endif
#if OUT_TILE_W() > 6
        vst1q_f32(output_oc1_base + 6  * CBLK(), v20);
#endif
#if OUT_TILE_W() > 7
        vst1q_f32(output_oc1_base + 7  * CBLK(), v21);
#endif
#if OUT_TILE_W() > 8
        vst1q_f32(output_oc1_base + 8  * CBLK(), v22);
#endif
#if OUT_TILE_W() > 9
        vst1q_f32(output_oc1_base + 9  * CBLK(), v23);
#endif
#if OUT_TILE_W() > 10
        vst1q_f32(output_oc1_base + 10 * CBLK(), v24);
#endif
#if OUT_TILE_W() > 11
        vst1q_f32(output_oc1_base + 11 * CBLK(), v25);
#endif
#if OUT_TILE_W() > 12
        vst1q_f32(output_oc1_base + 12 * CBLK(), v26);
#endif
#if OUT_TILE_W() > 13
        vst1q_f32(output_oc1_base + 13 * CBLK(), v27);
#endif
}

template<>
void ppl_kernel_arm_server_conv2d_fp32_conv_direct_ndarray_kernel<4, OUT_TILE_W()>(
    const float *input_base,
    const float *filter_base,
    const float *bias_base,
    float *output_base,
    float *sum_base,
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
    float32x4_t v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 ; (void)v0 ; (void)v1 ; (void)v2 ; (void)v3 ; (void)v4 ; (void)v5 ; (void)v6 ; (void)v7 ;
    float32x4_t v8 , v9 , v10, v11, v12, v13, v14, v15; (void)v8 ; (void)v9 ; (void)v10; (void)v11; (void)v12; (void)v13; (void)v14; (void)v15;
    float32x4_t v16, v17, v18, v19, v20, v21, v22, v23; (void)v16; (void)v17; (void)v18; (void)v19; (void)v20; (void)v21; (void)v22; (void)v23;
    float32x4_t v24, v25, v26, v27, v28, v29, v30, v31; (void)v24; (void)v25; (void)v26; (void)v27; (void)v28; (void)v29; (void)v30; (void)v31;

        if (nullptr == bias_base) {
            float *output_oc0_base = output_base;
#if OUT_TILE_W() > 0
            v0  = vld1q_f32(output_oc0_base + 0  * CBLK());
#endif
#if OUT_TILE_W() > 1
            v1  = vld1q_f32(output_oc0_base + 1  * CBLK());
#endif
#if OUT_TILE_W() > 2
            v2  = vld1q_f32(output_oc0_base + 2  * CBLK());
#endif
#if OUT_TILE_W() > 3
            v3  = vld1q_f32(output_oc0_base + 3  * CBLK());
#endif
#if OUT_TILE_W() > 4
            v4  = vld1q_f32(output_oc0_base + 4  * CBLK());
#endif
#if OUT_TILE_W() > 5
            v5  = vld1q_f32(output_oc0_base + 5  * CBLK());
#endif
#if OUT_TILE_W() > 6
            v6  = vld1q_f32(output_oc0_base + 6  * CBLK());
#endif
#if OUT_TILE_W() > 7
            v7  = vld1q_f32(output_oc0_base + 7  * CBLK());
#endif
#if OUT_TILE_W() > 8
            v8  = vld1q_f32(output_oc0_base + 8  * CBLK());
#endif
#if OUT_TILE_W() > 9
            v9  = vld1q_f32(output_oc0_base + 9  * CBLK());
#endif
#if OUT_TILE_W() > 10
            v10 = vld1q_f32(output_oc0_base + 10 * CBLK());
#endif
#if OUT_TILE_W() > 11
            v11 = vld1q_f32(output_oc0_base + 11 * CBLK());
#endif
#if OUT_TILE_W() > 12
            v12 = vld1q_f32(output_oc0_base + 12 * CBLK());
#endif
#if OUT_TILE_W() > 13
            v13 = vld1q_f32(output_oc0_base + 13 * CBLK());
#endif
        }  // load last output
        else {
#if OUT_TILE_W() > 0
            v0  = vld1q_f32(bias_base);
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

        const float *input_ic_base = input_base;
        int64_t ic = inC;
        const float *filter_ic_base = filter_base;
        do {
            const float *input_kh_base = input_ic_base;
            const float *filter_ptr = filter_ic_base;
            for (int64_t kh = 0; kh < fltH_valid; kh++) {
                const float *input_kw_base = input_kh_base;
                for (int64_t kw = 0; kw < fltW; kw++) {
                    const float *input_ptr = input_kw_base;
                    v28 = vld1q_f32(filter_ptr);
#if OUT_TILE_W() > 0
                    v30 = vld1q_dup_f32(input_ptr);
                    v0  = vfmaq_f32(v0 , v28, v30);
#endif
#if OUT_TILE_W() > 1
                    v31 = vld1q_dup_f32(input_ptr + strdW);
                    v1  = vfmaq_f32(v1 , v28, v31);
#endif
#if OUT_TILE_W() > 2
                    v30 = vld1q_dup_f32(input_ptr + 2 * strdW);
                    v2  = vfmaq_f32(v2 , v28, v30);
#endif
#if OUT_TILE_W() > 3
                    v31 = vld1q_dup_f32(input_ptr + 3 * strdW);
                    v3  = vfmaq_f32(v3 , v28, v31);
#endif
#if OUT_TILE_W() > 4
                    v30 = vld1q_dup_f32(input_ptr + 4 * strdW);
                    v4  = vfmaq_f32(v4 , v28, v30);
#endif
#if OUT_TILE_W() > 5
                    v31 = vld1q_dup_f32(input_ptr + 5 * strdW);
                    v5  = vfmaq_f32(v5 , v28, v31);
#endif
#if OUT_TILE_W() > 6
                    v30 = vld1q_dup_f32(input_ptr + 6 * strdW);
                    v6  = vfmaq_f32(v6 , v28, v30);
#endif
#if OUT_TILE_W() > 7
                    v31 = vld1q_dup_f32(input_ptr + 7 * strdW);
                    v7  = vfmaq_f32(v7 , v28, v31);
#endif
#if OUT_TILE_W() > 8
                    v30 = vld1q_dup_f32(input_ptr + 8 * strdW);
                    v8  = vfmaq_f32(v8 , v28, v30);
#endif
#if OUT_TILE_W() > 9
                    v31 = vld1q_dup_f32(input_ptr + 9 * strdW);
                    v9  = vfmaq_f32(v9 , v28, v31);
#endif
#if OUT_TILE_W() > 10
                    v30 = vld1q_dup_f32(input_ptr + 10 * strdW);
                    v10 = vfmaq_f32(v10, v28, v30);
#endif
#if OUT_TILE_W() > 11
                    v31 = vld1q_dup_f32(input_ptr + 11 * strdW);
                    v11 = vfmaq_f32(v11, v28, v31);
#endif
#if OUT_TILE_W() > 12
                    v30 = vld1q_dup_f32(input_ptr + 12 * strdW);
                    v12 = vfmaq_f32(v12, v28, v30);
#endif
#if OUT_TILE_W() > 13
                    v31 = vld1q_dup_f32(input_ptr + 13 * strdW);
                    v13 = vfmaq_f32(v13, v28, v31);
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
            v0  = vaddq_f32(v0 , vld1q_f32(sum_base + 0  * CBLK()));
#endif
#if OUT_TILE_W() > 1
            v1  = vaddq_f32(v1 , vld1q_f32(sum_base + 1  * CBLK()));
#endif
#if OUT_TILE_W() > 2
            v2  = vaddq_f32(v2 , vld1q_f32(sum_base + 2  * CBLK()));
#endif
#if OUT_TILE_W() > 3
            v3  = vaddq_f32(v3 , vld1q_f32(sum_base + 3  * CBLK()));
#endif
#if OUT_TILE_W() > 4
            v4  = vaddq_f32(v4 , vld1q_f32(sum_base + 4  * CBLK()));
#endif
#if OUT_TILE_W() > 5
            v5  = vaddq_f32(v5 , vld1q_f32(sum_base + 5  * CBLK()));
#endif
#if OUT_TILE_W() > 6
            v6  = vaddq_f32(v6 , vld1q_f32(sum_base + 6  * CBLK()));
#endif
#if OUT_TILE_W() > 7
            v7  = vaddq_f32(v7 , vld1q_f32(sum_base + 7  * CBLK()));
#endif
#if OUT_TILE_W() > 8
            v8  = vaddq_f32(v8 , vld1q_f32(sum_base + 8  * CBLK()));
#endif
#if OUT_TILE_W() > 9
            v9  = vaddq_f32(v9 , vld1q_f32(sum_base + 9  * CBLK()));
#endif
#if OUT_TILE_W() > 10
            v10 = vaddq_f32(v10, vld1q_f32(sum_base + 10 * CBLK()));
#endif
#if OUT_TILE_W() > 11
            v11 = vaddq_f32(v11, vld1q_f32(sum_base + 11 * CBLK()));
#endif
#if OUT_TILE_W() > 12
            v12 = vaddq_f32(v12, vld1q_f32(sum_base + 12 * CBLK()));
#endif
#if OUT_TILE_W() > 13
            v13 = vaddq_f32(v13, vld1q_f32(sum_base + 13 * CBLK()));
#endif            
        }
        if (fuse_type & conv_fuse_flag::RELU) { // relu
            v31 = vdupq_n_f32(0.0f);
#if OUT_TILE_W() > 0
            v0  = vmaxq_f32(v0 , v31);
#endif
#if OUT_TILE_W() > 1
            v1  = vmaxq_f32(v1 , v31);
#endif
#if OUT_TILE_W() > 2
            v2  = vmaxq_f32(v2 , v31);
#endif
#if OUT_TILE_W() > 3
            v3  = vmaxq_f32(v3 , v31);
#endif
#if OUT_TILE_W() > 4
            v4  = vmaxq_f32(v4 , v31);
#endif
#if OUT_TILE_W() > 5
            v5  = vmaxq_f32(v5 , v31);
#endif
#if OUT_TILE_W() > 6
            v6  = vmaxq_f32(v6 , v31);
#endif
#if OUT_TILE_W() > 7
            v7  = vmaxq_f32(v7 , v31);
#endif
#if OUT_TILE_W() > 8
            v8  = vmaxq_f32(v8 , v31);
#endif
#if OUT_TILE_W() > 9
            v9  = vmaxq_f32(v9 , v31);
#endif
#if OUT_TILE_W() > 10
            v10 = vmaxq_f32(v10, v31);
#endif
#if OUT_TILE_W() > 11
            v11 = vmaxq_f32(v11, v31);
#endif
#if OUT_TILE_W() > 12
            v12 = vmaxq_f32(v12, v31);
#endif
#if OUT_TILE_W() > 13
            v13 = vmaxq_f32(v13, v31);
#endif
        }
        if (fuse_type & conv_fuse_flag::RELU6) { // relu6
            v31 = vdupq_n_f32(6.0f);
#if OUT_TILE_W() > 0
            v0  = vminq_f32(v0 , v31);
#endif
#if OUT_TILE_W() > 1
            v1  = vminq_f32(v1 , v31);
#endif
#if OUT_TILE_W() > 2
            v2  = vminq_f32(v2 , v31);
#endif
#if OUT_TILE_W() > 3
            v3  = vminq_f32(v3 , v31);
#endif
#if OUT_TILE_W() > 4
            v4  = vminq_f32(v4 , v31);
#endif
#if OUT_TILE_W() > 5
            v5  = vminq_f32(v5 , v31);
#endif
#if OUT_TILE_W() > 6
            v6  = vminq_f32(v6 , v31);
#endif
#if OUT_TILE_W() > 7
            v7  = vminq_f32(v7 , v31);
#endif
#if OUT_TILE_W() > 8
            v8  = vminq_f32(v8 , v31);
#endif
#if OUT_TILE_W() > 9
            v9  = vminq_f32(v9 , v31);
#endif
#if OUT_TILE_W() > 10
            v10 = vminq_f32(v10, v31);
#endif
#if OUT_TILE_W() > 11
            v11 = vminq_f32(v11, v31);
#endif
#if OUT_TILE_W() > 12
            v12 = vminq_f32(v12, v31);
#endif
#if OUT_TILE_W() > 13
            v13 = vminq_f32(v13, v31);
#endif
        }

        float *output_oc0_base = output_base;
#if OUT_TILE_W() > 0
        vst1q_f32(output_oc0_base + 0  * CBLK(), v0 );
#endif
#if OUT_TILE_W() > 1
        vst1q_f32(output_oc0_base + 1  * CBLK(), v1 );
#endif
#if OUT_TILE_W() > 2
        vst1q_f32(output_oc0_base + 2  * CBLK(), v2 );
#endif
#if OUT_TILE_W() > 3
        vst1q_f32(output_oc0_base + 3  * CBLK(), v3 );
#endif
#if OUT_TILE_W() > 4
        vst1q_f32(output_oc0_base + 4  * CBLK(), v4 );
#endif
#if OUT_TILE_W() > 5
        vst1q_f32(output_oc0_base + 5  * CBLK(), v5 );
#endif
#if OUT_TILE_W() > 6
        vst1q_f32(output_oc0_base + 6  * CBLK(), v6 );
#endif
#if OUT_TILE_W() > 7
        vst1q_f32(output_oc0_base + 7  * CBLK(), v7 );
#endif
#if OUT_TILE_W() > 8
        vst1q_f32(output_oc0_base + 8  * CBLK(), v8 );
#endif
#if OUT_TILE_W() > 9
        vst1q_f32(output_oc0_base + 9  * CBLK(), v9 );
#endif
#if OUT_TILE_W() > 10
        vst1q_f32(output_oc0_base + 10 * CBLK(), v10);
#endif
#if OUT_TILE_W() > 11
        vst1q_f32(output_oc0_base + 11 * CBLK(), v11);
#endif
#if OUT_TILE_W() > 12
        vst1q_f32(output_oc0_base + 12 * CBLK(), v12);
#endif
#if OUT_TILE_W() > 13
        vst1q_f32(output_oc0_base + 13 * CBLK(), v13);
#endif
}