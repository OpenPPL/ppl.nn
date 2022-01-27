.align 4

#define RVV_0_7_1
#ifdef RVV_0_7_1
    #define vle8        vlb
    #define vle16       vlh
    #define vle32       vlw

    #define vse8        vsb
    #define vse16       vsh
    #define vse32       vsw
#endif

.macro PPL_CONV_GEMM_KERNEL_M8N4K1 ak0 bi cn0 cn1 cn2 cn3
    vrgather.vi    v4, v0, \bi
    vrgather.vi    v5, v1, \bi
    vfmacc.vv      \cn0, v4, \ak0
    vrgather.vi    v6, v2, \bi
    vfmacc.vv      \cn1, v5, \ak0
    vrgather.vi    v7, v3, \bi

    vfmacc.vv      \cn2, v6, \ak0
    vfmacc.vv      \cn3, v7, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N4K1_FIRST ak0 bi cn0 cn1 cn2 cn3
    vrgather.vi    v4, v0, \bi
    vrgather.vi    v5, v1, \bi
    vfmul.vv      \cn0, v4, \ak0
    vrgather.vi    v6, v2, \bi
    vfmul.vv      \cn1, v5, \ak0
    vrgather.vi    v7, v3, \bi

    vfmul.vv      \cn2, v6, \ak0
    vfmul.vv      \cn3, v7, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N3K1 ak0 bi cn0 cn1 cn2
    vrgather.vi    v4, v0, \bi
    vrgather.vi    v5, v1, \bi
    vrgather.vi    v6, v2, \bi

    vfmacc.vv      \cn0, v4, \ak0
    vfmacc.vv      \cn1, v5, \ak0
    vfmacc.vv      \cn2, v6, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N3K1_FIRST ak0 bi cn0 cn1 cn2
    vrgather.vi    v4, v0, \bi
    vrgather.vi    v5, v1, \bi
    vrgather.vi    v6, v2, \bi

    vfmul.vv      \cn0, v4, \ak0
    vfmul.vv      \cn1, v5, \ak0
    vfmul.vv      \cn2, v6, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N2K1 ak0 bi cn0 cn1
    vrgather.vi    v4, v0, \bi
    vrgather.vi    v5, v1, \bi

    vfmacc.vv      \cn0, v4, \ak0
    vfmacc.vv      \cn1, v5, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N2K1_FIRST ak0 bi cn0 cn1
    vrgather.vi    v4, v0, \bi
    vrgather.vi    v5, v1, \bi

    vfmul.vv      \cn0, v4, \ak0
    vfmul.vv      \cn1, v5, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N1K1 ak0 bi cn0
    vrgather.vi    v4, v0, \bi
    vfmacc.vv      \cn0, v4, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N1K1_FIRST ak0 bi cn0
    vrgather.vi    v4, v0, \bi
    vfmul.vv      \cn0, v4, \ak0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N4K8_EXCEPT_FIRST cn0 cn1 cn2 cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v9  1 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v10 2 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v11 3 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v12 4 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v13 5 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v14 6 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v15 7 \cn0 \cn1 \cn2 \cn3
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST cn0 cn1 cn2 cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1_FIRST v8 0 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K8_EXCEPT_FIRST \cn0 \cn1 \cn2 \cn3
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N4K8 cn0 cn1 cn2 cn3
    PPL_CONV_GEMM_KERNEL_M8N4K1 v8  0 \cn0 \cn1 \cn2 \cn3
    PPL_CONV_GEMM_KERNEL_M8N4K8_EXCEPT_FIRST \cn0 \cn1 \cn2 \cn3
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N3K8_EXCEPT_FIRST cn0 cn1 cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v9  1 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v10 2 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v11 3 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v12 4 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v13 5 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v14 6 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v15 7 \cn0 \cn1 \cn2
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N3K8_FIRST cn0 cn1 cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1_FIRST v8 0 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K8_EXCEPT_FIRST \cn0 \cn1 \cn2
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N3K8 cn0 cn1 cn2
    PPL_CONV_GEMM_KERNEL_M8N3K1 v8  0 \cn0 \cn1 \cn2
    PPL_CONV_GEMM_KERNEL_M8N3K8_EXCEPT_FIRST \cn0 \cn1 \cn2
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N2K8_EXCEPT_FIRST cn0 cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v9  1 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v10 2 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v11 3 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v12 4 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v13 5 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v14 6 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v15 7 \cn0 \cn1
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N2K8_FIRST cn0 cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1_FIRST v8 0 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K8_EXCEPT_FIRST \cn0 \cn1
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N2K8 cn0 cn1
    PPL_CONV_GEMM_KERNEL_M8N2K1 v8  0 \cn0 \cn1
    PPL_CONV_GEMM_KERNEL_M8N2K8_EXCEPT_FIRST \cn0 \cn1
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N1K8_EXCEPT_FIRST cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v9  1 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v10 2 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v11 3 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v12 4 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v13 5 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v14 6 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v15 7 \cn0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N1K8_FIRST cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1_FIRST v8 0 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K8_EXCEPT_FIRST \cn0
.endm

.macro PPL_CONV_GEMM_KERNEL_M8N1K8 cn0
    PPL_CONV_GEMM_KERNEL_M8N1K1 v8  0 \cn0
    PPL_CONV_GEMM_KERNEL_M8N1K8_EXCEPT_FIRST \cn0
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 a_ptr
    vle16.v        v8, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
    vle16.v        v9, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
    vle16.v        v10, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
    vle16.v        v11, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
    vle16.v        v12, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
    vle16.v        v13, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
    vle16.v        v14, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
    vle16.v        v15, (\a_ptr)
    addi           \a_ptr, \a_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 b_ptr
    vle16.v        v0, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
    vle16.v        v1, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
    vle16.v        v2, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
    vle16.v        v3, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 b_ptr
    vle16.v        v0, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
    vle16.v        v1, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
    vle16.v        v2, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 b_ptr
    vle16.v        v0, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
    vle16.v        v1, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 b_ptr
    vle16.v        v0, (\b_ptr)
    addi           \b_ptr, \b_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N16 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v29, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v30, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v31, (\c_ptr)
    addi           \c_ptr, \c_ptr, -240
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N15 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v29, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v30, (\c_ptr)
    addi           \c_ptr, \c_ptr, -224
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N14 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v29, (\c_ptr)
    addi           \c_ptr, \c_ptr, -208
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N13 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, -192
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N12 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, -176
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N11 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, -160
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N10 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, -144
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N9 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, -128
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N8 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, -112
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N7 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, -96
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N6 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, -80
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N5 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, -64
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N4 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, -48
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N3 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, -32
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N2 c_ptr
    vle16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vle16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, -16
.endm

.macro PPL_CONV_GEMM_KERNEL_LOAD_C_M8N1 c_ptr
    vle16.v        v16, (\c_ptr)
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N16 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v29, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v30, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v31, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N15 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v29, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v30, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N14 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v29, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N13 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v28, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N12 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v27, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N11 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v26, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N10 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v25, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N9 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v24, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N8 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v23, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N7 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v22, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N6 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v21, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N5 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v20, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N4 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v19, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N3 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v18, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N2 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
    vse16.v        v17, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_STORE_C_M8N1 c_ptr
    vse16.v        v16, (\c_ptr)
    addi           \c_ptr, \c_ptr, 16
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N16_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v28 v29 v30 v31
.endm

.macro gemm_common_kernel_m8n16 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N16 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N16_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N16 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N16_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v28 v29 v30 v31
.endm

.macro gemm_common_kernel_m8n16_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N16 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N16_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N16_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N16 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N15_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8 v28 v29 v30
.endm

.macro gemm_common_kernel_m8n15 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N15 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N15_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N15 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N15_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8_FIRST v28 v29 v30
.endm

.macro gemm_common_kernel_m8n15_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N15 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N15_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N15_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N15 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N14_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8 v28 v29
.endm

.macro gemm_common_kernel_m8n14 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N14 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N14_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N14 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N14_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8_FIRST v28 v29
.endm

.macro gemm_common_kernel_m8n14_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N14 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N14_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N14_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N14 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N13_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8 v28
.endm

.macro gemm_common_kernel_m8n13 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N13 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N13_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N13 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N13_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v24 v25 v26 v27

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8_FIRST v28
.endm

.macro gemm_common_kernel_m8n13_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N13 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N13_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N13_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N13 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N12_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v24 v25 v26 v27
.endm

.macro gemm_common_kernel_m8n12 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N12 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N12_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N12 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N12_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v24 v25 v26 v27
.endm

.macro gemm_common_kernel_m8n12_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N12 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N12_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N12_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N12 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N11_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8 v24 v25 v26
.endm

.macro gemm_common_kernel_m8n11 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N11 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N11_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N11 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N11_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8_FIRST v24 v25 v26
.endm

.macro gemm_common_kernel_m8n11_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N11 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N11_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N11_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N11 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N10_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8 v24 v25
.endm

.macro gemm_common_kernel_m8n10 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N10 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N10_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N10 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N10_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8_FIRST v24 v25
.endm

.macro gemm_common_kernel_m8n10_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N10 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N10_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N10_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N10 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N9_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8 v24
.endm

.macro gemm_common_kernel_m8n9 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N9 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N9_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N9 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N9_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8_FIRST v24
.endm

.macro gemm_common_kernel_m8n9_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N9 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N9_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N9_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N9 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N8_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v20 v21 v22 v23
.endm

.macro gemm_common_kernel_m8n8 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N8 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N8_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N8 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N8_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v20 v21 v22 v23
.endm

.macro gemm_common_kernel_m8n8_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N8 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N8_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N8_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N8 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N7_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8 v20 v21 v22
.endm

.macro gemm_common_kernel_m8n7 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N7 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N7_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N7 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N7_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8_FIRST v20 v21 v22
.endm

.macro gemm_common_kernel_m8n7_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N7 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N7_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N7_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N7 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N6_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8 v20 v21
.endm

.macro gemm_common_kernel_m8n6 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N6 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N6_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N6 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N6_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8_FIRST v20 v21
.endm

.macro gemm_common_kernel_m8n6_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N6 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N6_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N6_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N6 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N5_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8 v20
.endm

.macro gemm_common_kernel_m8n5 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N5 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N5_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N5 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N5_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8_FIRST v20
.endm

.macro gemm_common_kernel_m8n5_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N5 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N5_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N5_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N5 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N4_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8 v16 v17 v18 v19
.endm

.macro gemm_common_kernel_m8n4 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N4 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N4_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N4 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N4_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N4K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N4K8_FIRST v16 v17 v18 v19
.endm

.macro gemm_common_kernel_m8n4_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N4 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N4_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N4_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N4 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N3_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8 v16 v17 v18
.endm

.macro gemm_common_kernel_m8n3 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N3 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N3_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N3 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N3_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N3K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N3K8_FIRST v16 v17 v18
.endm

.macro gemm_common_kernel_m8n3_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N3 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N3_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N3_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N3 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N2_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8 v16 v17
.endm

.macro gemm_common_kernel_m8n2 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N2 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N2_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N2 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N2_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N2K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N2K8_FIRST v16 v17
.endm

.macro gemm_common_kernel_m8n2_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N2 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N2_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N2_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N2 \c_loc
.endm


.macro PPL_CONV_GEMM_KERNEL_FP16_M8N1_CORE a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8 v16
.endm

.macro gemm_common_kernel_m8n1 a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N1 \c_loc
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N1_CORE \a_loc \b_loc


    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N1 \c_loc
.endm

.macro PPL_CONV_GEMM_KERNEL_FP16_M8N1_CORE_FIRST a_loc b_loc
    PPL_CONV_GEMM_KERNEL_LOAD_A_M8K8 \a_loc

    PPL_CONV_GEMM_KERNEL_LOAD_B_N1K8 \b_loc
    PPL_CONV_GEMM_KERNEL_M8N1K8_FIRST v16
.endm

.macro gemm_common_kernel_m8n1_first a_loc b_loc c_loc b_stride k
0:
    PPL_CONV_GEMM_KERNEL_LOAD_C_M8N1 \c_loc
    PPL_CONV_GEMM_KERNEL_FP16_M8N1_CORE_FIRST \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    beq             \k, zero, 2f
1:
    PPL_CONV_GEMM_KERNEL_FP16_M8N1_CORE \a_loc \b_loc

    add             \b_loc, \b_loc, \b_stride
    addi            \k, \k, -8
    bnez            \k, 1b

2:
    PPL_CONV_GEMM_KERNEL_STORE_C_M8N1 \c_loc
.endm

.macro this_preserve_caller
    addi            sp, sp, -576
    sd              t0, 0(sp)
    sd              t1, 8(sp)
    sd              t2, 16(sp)
    sd              t3, 24(sp)
    sd              t4, 32(sp)
    sd              s2, 40(sp)
    sd              s3, 48(sp)
    sd              s4, 56(sp)

    addi            t1, zero, 8
    vsetvli         t0, t1, e16

    addi            t0, sp, 64
    vse16.v         v0, (t0)
    addi            t0, t0, 16
    vse16.v         v1, (t0)
    addi            t0, t0, 16
    vse16.v         v2, (t0)
    addi            t0, t0, 16
    vse16.v         v3, (t0)
    addi            t0, t0, 16
    vse16.v         v4, (t0)
    addi            t0, t0, 16
    vse16.v         v5, (t0)
    addi            t0, t0, 16
    vse16.v         v6, (t0)
    addi            t0, t0, 16
    vse16.v         v7, (t0)
    addi            t0, t0, 16
    vse16.v         v8, (t0)
    addi            t0, t0, 16
    vse16.v         v9, (t0)
    addi            t0, t0, 16
    vse16.v         v10, (t0)
    addi            t0, t0, 16
    vse16.v         v11, (t0)
    addi            t0, t0, 16
    vse16.v         v12, (t0)
    addi            t0, t0, 16
    vse16.v         v13, (t0)
    addi            t0, t0, 16
    vse16.v         v14, (t0)
    addi            t0, t0, 16
    vse16.v         v15, (t0)
    addi            t0, t0, 16
    vse16.v         v16, (t0)
    addi            t0, t0, 16
    vse16.v         v17, (t0)
    addi            t0, t0, 16
    vse16.v         v18, (t0)
    addi            t0, t0, 16
    vse16.v         v19, (t0)
    addi            t0, t0, 16
    vse16.v         v20, (t0)
    addi            t0, t0, 16
    vse16.v         v21, (t0)
    addi            t0, t0, 16
    vse16.v         v22, (t0)
    addi            t0, t0, 16
    vse16.v         v23, (t0)
    addi            t0, t0, 16
    vse16.v         v24, (t0)
    addi            t0, t0, 16
    vse16.v         v25, (t0)
    addi            t0, t0, 16
    vse16.v         v26, (t0)
    addi            t0, t0, 16
    vse16.v         v27, (t0)
    addi            t0, t0, 16
    vse16.v         v28, (t0)
    addi            t0, t0, 16
    vse16.v         v29, (t0)
    addi            t0, t0, 16
    vse16.v         v30, (t0)
    addi            t0, t0, 16
    vse16.v         v31, (t0)
.endm

.macro this_restore_caller
    ld              t1, 8(sp)
    ld              t2, 16(sp)
    ld              t3, 24(sp)
    ld              t4, 32(sp)
    ld              s2, 40(sp)
    ld              s3, 48(sp)
    ld              s4, 56(sp)

    addi            t0, sp, 64
    vle16.v         v0, (t0)
    addi            t0, t0, 16
    vle16.v         v1, (t0)
    addi            t0, t0, 16
    vle16.v         v2, (t0)
    addi            t0, t0, 16
    vle16.v         v3, (t0)
    addi            t0, t0, 16
    vle16.v         v4, (t0)
    addi            t0, t0, 16
    vle16.v         v5, (t0)
    addi            t0, t0, 16
    vle16.v         v6, (t0)
    addi            t0, t0, 16
    vle16.v         v7, (t0)
    addi            t0, t0, 16
    vle16.v         v8, (t0)
    addi            t0, t0, 16
    vle16.v         v9, (t0)
    addi            t0, t0, 16
    vle16.v         v10, (t0)
    addi            t0, t0, 16
    vle16.v         v11, (t0)
    addi            t0, t0, 16
    vle16.v         v12, (t0)
    addi            t0, t0, 16
    vle16.v         v13, (t0)
    addi            t0, t0, 16
    vle16.v         v14, (t0)
    addi            t0, t0, 16
    vle16.v         v15, (t0)
    addi            t0, t0, 16
    vle16.v         v16, (t0)
    addi            t0, t0, 16
    vle16.v         v17, (t0)
    addi            t0, t0, 16
    vle16.v         v18, (t0)
    addi            t0, t0, 16
    vle16.v         v19, (t0)
    addi            t0, t0, 16
    vle16.v         v20, (t0)
    addi            t0, t0, 16
    vle16.v         v21, (t0)
    addi            t0, t0, 16
    vle16.v         v22, (t0)
    addi            t0, t0, 16
    vle16.v         v23, (t0)
    addi            t0, t0, 16
    vle16.v         v24, (t0)
    addi            t0, t0, 16
    vle16.v         v25, (t0)
    addi            t0, t0, 16
    vle16.v         v26, (t0)
    addi            t0, t0, 16
    vle16.v         v27, (t0)
    addi            t0, t0, 16
    vle16.v         v28, (t0)
    addi            t0, t0, 16
    vle16.v         v29, (t0)
    addi            t0, t0, 16
    vle16.v         v30, (t0)
    addi            t0, t0, 16
    vle16.v         v31, (t0)

    ld              t0, 0(sp)
    addi            sp, sp, 576
.endm

#define a_loc   a0
#define b_loc   a1
#define c_loc   a2
#define m       a3
#define n       a4
#define k       a5

#define loop_n      t0
#define loop_k      t1
#define a_stride    t2
#define b_stride    t3
#define b_left_stride t4

#define b_loc_d     s2
#define a_core_loc  s3
#define b_core_loc  s4

.type gemm_common_m8n16_left0_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left0_rv64_fp16
.hidden gemm_common_m8n16_left0_rv64_fp16

gemm_common_m8n16_left0_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16

.ppl_conv_gemm_fp16_m8n16_left0_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_stride, b_stride, -256

.ppl_conv_gemm_fp16_m8n16_left0_loop1:
    mv              loop_n, n                   // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

.ppl_conv_gemm_fp16_m8n16_left0_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    bnez            loop_n, .ppl_conv_gemm_fp16_m8n16_left0_loop2
    // .ppl_conv_gemm_fp16_m8n16_left0_loop2

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left0_loop1
    // .ppl_conv_gemm_fp16_m8n16_left0_loop1

    this_restore_caller
    ret


.type gemm_common_m8n16_left0_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left0_first_rv64_fp16
.hidden gemm_common_m8n16_left0_first_rv64_fp16

gemm_common_m8n16_left0_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16

.ppl_conv_gemm_fp16_m8n16_left0_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_stride, b_stride, -256

.ppl_conv_gemm_fp16_m8n16_left0_first_loop1:
    mv              loop_n, n                   // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

.ppl_conv_gemm_fp16_m8n16_left0_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    bnez            loop_n, .ppl_conv_gemm_fp16_m8n16_left0_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left0_first_loop2

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left0_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left0_first_loop1

    this_restore_caller
    ret


.type gemm_common_m8n16_left15_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left15_first_rv64_fp16
.hidden gemm_common_m8n16_left15_first_rv64_fp16

gemm_common_m8n16_left15_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left15_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -240
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left15_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left15_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left15_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left15_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left15_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left15_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n15_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 240       // b_loc += 15(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left15_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left15_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left15_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left15_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left15_rv64_fp16
.hidden gemm_common_m8n16_left15_rv64_fp16

gemm_common_m8n16_left15_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left15_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -240
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left15_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left15_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left15_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left15_loop2
    // .ppl_conv_gemm_fp16_m8n16_left15_loop2    

.ppl_conv_gemm_fp16_m8n16_left15_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n15 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 240       // b_loc += 15(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left15_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left15_loop1
    // .ppl_conv_gemm_fp16_m8n16_left15_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left14_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left14_first_rv64_fp16
.hidden gemm_common_m8n16_left14_first_rv64_fp16

gemm_common_m8n16_left14_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left14_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -224
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left14_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left14_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left14_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left14_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left14_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left14_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n14_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 224       // b_loc += 14(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left14_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left14_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left14_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left14_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left14_rv64_fp16
.hidden gemm_common_m8n16_left14_rv64_fp16

gemm_common_m8n16_left14_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left14_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -224
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left14_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left14_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left14_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left14_loop2
    // .ppl_conv_gemm_fp16_m8n16_left14_loop2    

.ppl_conv_gemm_fp16_m8n16_left14_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n14 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 224       // b_loc += 14(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left14_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left14_loop1
    // .ppl_conv_gemm_fp16_m8n16_left14_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left13_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left13_first_rv64_fp16
.hidden gemm_common_m8n16_left13_first_rv64_fp16

gemm_common_m8n16_left13_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left13_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -208
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left13_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left13_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left13_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left13_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left13_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left13_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n13_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 208       // b_loc += 13(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left13_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left13_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left13_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left13_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left13_rv64_fp16
.hidden gemm_common_m8n16_left13_rv64_fp16

gemm_common_m8n16_left13_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left13_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -208
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left13_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left13_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left13_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left13_loop2
    // .ppl_conv_gemm_fp16_m8n16_left13_loop2    

.ppl_conv_gemm_fp16_m8n16_left13_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n13 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 208       // b_loc += 13(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left13_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left13_loop1
    // .ppl_conv_gemm_fp16_m8n16_left13_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left12_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left12_first_rv64_fp16
.hidden gemm_common_m8n16_left12_first_rv64_fp16

gemm_common_m8n16_left12_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left12_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -192
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left12_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left12_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left12_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left12_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left12_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left12_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n12_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 192       // b_loc += 12(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left12_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left12_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left12_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left12_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left12_rv64_fp16
.hidden gemm_common_m8n16_left12_rv64_fp16

gemm_common_m8n16_left12_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left12_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -192
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left12_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left12_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left12_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left12_loop2
    // .ppl_conv_gemm_fp16_m8n16_left12_loop2    

.ppl_conv_gemm_fp16_m8n16_left12_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n12 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 192       // b_loc += 12(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left12_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left12_loop1
    // .ppl_conv_gemm_fp16_m8n16_left12_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left11_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left11_first_rv64_fp16
.hidden gemm_common_m8n16_left11_first_rv64_fp16

gemm_common_m8n16_left11_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left11_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -176
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left11_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left11_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left11_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left11_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left11_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left11_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n11_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 176       // b_loc += 11(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left11_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left11_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left11_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left11_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left11_rv64_fp16
.hidden gemm_common_m8n16_left11_rv64_fp16

gemm_common_m8n16_left11_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left11_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -176
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left11_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left11_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left11_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left11_loop2
    // .ppl_conv_gemm_fp16_m8n16_left11_loop2    

.ppl_conv_gemm_fp16_m8n16_left11_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n11 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 176       // b_loc += 11(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left11_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left11_loop1
    // .ppl_conv_gemm_fp16_m8n16_left11_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left10_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left10_first_rv64_fp16
.hidden gemm_common_m8n16_left10_first_rv64_fp16

gemm_common_m8n16_left10_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left10_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -160
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left10_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left10_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left10_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left10_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left10_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left10_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n10_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 160       // b_loc += 10(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left10_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left10_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left10_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left10_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left10_rv64_fp16
.hidden gemm_common_m8n16_left10_rv64_fp16

gemm_common_m8n16_left10_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left10_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -160
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left10_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left10_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left10_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left10_loop2
    // .ppl_conv_gemm_fp16_m8n16_left10_loop2    

.ppl_conv_gemm_fp16_m8n16_left10_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n10 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 160       // b_loc += 10(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left10_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left10_loop1
    // .ppl_conv_gemm_fp16_m8n16_left10_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left9_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left9_first_rv64_fp16
.hidden gemm_common_m8n16_left9_first_rv64_fp16

gemm_common_m8n16_left9_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left9_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -144
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left9_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left9_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left9_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left9_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left9_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left9_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n9_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 144       // b_loc += 9(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left9_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left9_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left9_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left9_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left9_rv64_fp16
.hidden gemm_common_m8n16_left9_rv64_fp16

gemm_common_m8n16_left9_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left9_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -144
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left9_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left9_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left9_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left9_loop2
    // .ppl_conv_gemm_fp16_m8n16_left9_loop2    

.ppl_conv_gemm_fp16_m8n16_left9_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n9 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 144       // b_loc += 9(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left9_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left9_loop1
    // .ppl_conv_gemm_fp16_m8n16_left9_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left8_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left8_first_rv64_fp16
.hidden gemm_common_m8n16_left8_first_rv64_fp16

gemm_common_m8n16_left8_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left8_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -128
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left8_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left8_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left8_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left8_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left8_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left8_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n8_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 128       // b_loc += 8(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left8_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left8_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left8_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left8_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left8_rv64_fp16
.hidden gemm_common_m8n16_left8_rv64_fp16

gemm_common_m8n16_left8_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left8_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -128
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left8_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left8_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left8_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left8_loop2
    // .ppl_conv_gemm_fp16_m8n16_left8_loop2    

.ppl_conv_gemm_fp16_m8n16_left8_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n8 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 128       // b_loc += 8(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left8_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left8_loop1
    // .ppl_conv_gemm_fp16_m8n16_left8_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left7_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left7_first_rv64_fp16
.hidden gemm_common_m8n16_left7_first_rv64_fp16

gemm_common_m8n16_left7_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left7_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -112
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left7_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left7_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left7_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left7_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left7_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left7_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n7_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 112       // b_loc += 7(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left7_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left7_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left7_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left7_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left7_rv64_fp16
.hidden gemm_common_m8n16_left7_rv64_fp16

gemm_common_m8n16_left7_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left7_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -112
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left7_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left7_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left7_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left7_loop2
    // .ppl_conv_gemm_fp16_m8n16_left7_loop2    

.ppl_conv_gemm_fp16_m8n16_left7_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n7 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 112       // b_loc += 7(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left7_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left7_loop1
    // .ppl_conv_gemm_fp16_m8n16_left7_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left6_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left6_first_rv64_fp16
.hidden gemm_common_m8n16_left6_first_rv64_fp16

gemm_common_m8n16_left6_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left6_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -96
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left6_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left6_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left6_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left6_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left6_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left6_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n6_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 96       // b_loc += 6(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left6_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left6_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left6_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left6_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left6_rv64_fp16
.hidden gemm_common_m8n16_left6_rv64_fp16

gemm_common_m8n16_left6_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left6_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -96
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left6_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left6_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left6_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left6_loop2
    // .ppl_conv_gemm_fp16_m8n16_left6_loop2    

.ppl_conv_gemm_fp16_m8n16_left6_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n6 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 96       // b_loc += 6(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left6_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left6_loop1
    // .ppl_conv_gemm_fp16_m8n16_left6_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left5_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left5_first_rv64_fp16
.hidden gemm_common_m8n16_left5_first_rv64_fp16

gemm_common_m8n16_left5_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left5_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -80
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left5_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left5_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left5_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left5_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left5_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left5_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n5_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 80       // b_loc += 5(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left5_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left5_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left5_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left5_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left5_rv64_fp16
.hidden gemm_common_m8n16_left5_rv64_fp16

gemm_common_m8n16_left5_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left5_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -80
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left5_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left5_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left5_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left5_loop2
    // .ppl_conv_gemm_fp16_m8n16_left5_loop2    

.ppl_conv_gemm_fp16_m8n16_left5_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n5 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 80       // b_loc += 5(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left5_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left5_loop1
    // .ppl_conv_gemm_fp16_m8n16_left5_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left4_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left4_first_rv64_fp16
.hidden gemm_common_m8n16_left4_first_rv64_fp16

gemm_common_m8n16_left4_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left4_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -64
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left4_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left4_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left4_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left4_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left4_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left4_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n4_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 64       // b_loc += 4(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left4_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left4_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left4_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left4_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left4_rv64_fp16
.hidden gemm_common_m8n16_left4_rv64_fp16

gemm_common_m8n16_left4_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left4_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -64
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left4_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left4_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left4_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left4_loop2
    // .ppl_conv_gemm_fp16_m8n16_left4_loop2    

.ppl_conv_gemm_fp16_m8n16_left4_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n4 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 64       // b_loc += 4(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left4_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left4_loop1
    // .ppl_conv_gemm_fp16_m8n16_left4_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left3_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left3_first_rv64_fp16
.hidden gemm_common_m8n16_left3_first_rv64_fp16

gemm_common_m8n16_left3_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left3_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -48
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left3_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left3_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left3_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left3_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left3_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left3_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n3_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 48       // b_loc += 3(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left3_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left3_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left3_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left3_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left3_rv64_fp16
.hidden gemm_common_m8n16_left3_rv64_fp16

gemm_common_m8n16_left3_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left3_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -48
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left3_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left3_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left3_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left3_loop2
    // .ppl_conv_gemm_fp16_m8n16_left3_loop2    

.ppl_conv_gemm_fp16_m8n16_left3_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n3 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 48       // b_loc += 3(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left3_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left3_loop1
    // .ppl_conv_gemm_fp16_m8n16_left3_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left2_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left2_first_rv64_fp16
.hidden gemm_common_m8n16_left2_first_rv64_fp16

gemm_common_m8n16_left2_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left2_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -32
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left2_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left2_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left2_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left2_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left2_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left2_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n2_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 32       // b_loc += 2(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left2_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left2_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left2_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left2_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left2_rv64_fp16
.hidden gemm_common_m8n16_left2_rv64_fp16

gemm_common_m8n16_left2_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left2_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -32
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left2_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left2_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left2_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left2_loop2
    // .ppl_conv_gemm_fp16_m8n16_left2_loop2    

.ppl_conv_gemm_fp16_m8n16_left2_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n2 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 32       // b_loc += 2(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left2_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left2_loop1
    // .ppl_conv_gemm_fp16_m8n16_left2_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left1_first_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left1_first_rv64_fp16
.hidden gemm_common_m8n16_left1_first_rv64_fp16

gemm_common_m8n16_left1_first_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left1_first_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -16
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left1_first_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left1_first_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left1_first_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16_first a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left1_first_loop2
    // .ppl_conv_gemm_fp16_m8n16_left1_first_loop2    

.ppl_conv_gemm_fp16_m8n16_left1_first_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n1_first a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 16       // b_loc += 1(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left1_first_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left1_first_loop1
    // .ppl_conv_gemm_fp16_m8n16_left1_first_loop1    

    this_restore_caller
    ret    


.type gemm_common_m8n16_left1_rv64_fp16 STT_FUNC
.global gemm_common_m8n16_left1_rv64_fp16
.hidden gemm_common_m8n16_left1_rv64_fp16

gemm_common_m8n16_left1_rv64_fp16:
    this_preserve_caller
    addi            t1, zero, 8
    vsetvli         t0, t1, e16
    
.ppl_conv_gemm_fp16_m8n16_left1_init:
    slli            a_stride, k, 4              // a_stride = k * 8(m) * sizeof(__fp16)
    slli            b_stride, n, 4              // b_stride = n * 8(k) * sizeof(__fp16)
    addi            b_left_stride, b_stride, -16
    addi            b_stride, b_stride, -256    

.ppl_conv_gemm_fp16_m8n16_left1_loop1:
    addi            loop_n, n, -16               // load n
    mv              b_loc_d, b_loc
    addi            m, m, -8

    bge             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left1_loop2_left    

.ppl_conv_gemm_fp16_m8n16_left1_loop2:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d
    addi            loop_n, loop_n, -16

    gemm_common_kernel_m8n16 a_core_loc b_core_loc c_loc b_stride loop_k

    addi            b_loc_d, b_loc_d, 256       // b_loc += 16(n) * 8(k) * sizeof(__fp16)
    blt             zero, loop_n, .ppl_conv_gemm_fp16_m8n16_left1_loop2
    // .ppl_conv_gemm_fp16_m8n16_left1_loop2    

.ppl_conv_gemm_fp16_m8n16_left1_loop2_left:
    mv              loop_k, k
    mv              a_core_loc, a_loc
    mv              b_core_loc, b_loc_d

    gemm_common_kernel_m8n1 a_core_loc b_core_loc c_loc b_left_stride loop_k

    addi            b_loc_d, b_loc_d, 16       // b_loc += 1(n) * 8(k) * sizeof(__fp16)
    // .ppl_conv_gemm_fp16_m8n16_left1_loop2_left

    add             a_loc, a_loc, a_stride
    bnez            m, .ppl_conv_gemm_fp16_m8n16_left1_loop1
    // .ppl_conv_gemm_fp16_m8n16_left1_loop1    

    this_restore_caller
    ret

