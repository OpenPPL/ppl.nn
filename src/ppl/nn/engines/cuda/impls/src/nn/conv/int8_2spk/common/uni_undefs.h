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

////////////////////////////////////////
// kernel list macros
////////////////////////////////////////

#undef SPK_KPARAM_LIST
#undef TOTAL_KPARAM_LIST

////////////////////////////////////////
// customized macros
////////////////////////////////////////

#undef TILE_N_PER_CTA
#undef TILE_M_PER_CTA

#undef TILE_K_PER_CTA
#undef TILE_K_PER_SET
#undef TILE_K_PER_WARP

#undef TILE_N_PER_WARP
#undef TILE_M_PER_WARP

#undef KERNEL_NAME

////////////////////////////////////////
// align functions
////////////////////////////////////////

#undef Align
#undef DivUp

#undef Min
#undef Max

////////////////////////////////////////
// boundary check
////////////////////////////////////////

#undef WidthInRange
#undef HeightInRange

////////////////////////////////////////
// constant cta size macros
////////////////////////////////////////

#undef _4CHAR_TO_INT_
#undef _4INT_TO_INT4_
#undef _2INT_TO_INT2_

#undef _2HALF_TO_INT_
#undef _2INT2_TO_INT4_

#undef _C1_
#undef _C2_
#undef _C4_
#undef _C8_
#undef _C16_
#undef _C32_

#undef _1INT_
#undef _2INT_
#undef _4INT_
#undef _8INT_

#undef _1INT4_
#undef _2INT4_
#undef _4INT4_
#undef _8INT4_

#undef _1INT8_
#undef _2INT8_
#undef _4INT8_
#undef _8INT8_

#undef _1HALF_
#undef _2HALF_
#undef _4HALF_
#undef _8HALF_

#undef _1HALF2_
#undef _2HALF2_
#undef _4HALF2_
#undef _8HALF2_

#undef _1MMA_
#undef _2MMA_
#undef _4MMA_
#undef _8MMA_

#undef _HALF_ZERO_

#undef _INT_TO_BYTE_
#undef _INT_TO_2HALF_
#undef _INT2_TO_2HALF2_
#undef _INT2_TO_2INT_

#undef _INT4_TO_INT4_
#undef _INT4_TO_2INT2_
#undef _INT4_TO_4INT_
#undef _INT4_TO_4HALF2_
#undef _INT4_TO_8HALF_

#undef SMEM_ROW_V1_SIZE
#undef SMEM_ROW_V4_SIZE
#undef SMEM_ROW_BYTE_SIZE
#undef SMEM_ROW_BIT_SIZE


////////////////////////////////////////
// mma size macros
////////////////////////////////////////

#undef TILE_M_PER_MMA
#undef TILE_K_PER_MMA
#undef TILE_N_PER_MMA
#undef TILE_M_PER_MMA_HALF

#undef MMA_SIZE_Y_IN_THD
#undef MMA_SIZE_Y_IN_THD

#undef MMA_SIZE_X_IN_BITS

////////////////////////////////////////
// thread / warp / cta size macros
////////////////////////////////////////

#undef WARP_SIZE_IN_THD
#undef WARP_SIZE_IN_BITS

#undef WARP_SIZE_X_IN_THD
#undef WARP_SIZE_Y_IN_THD

#undef SET_SIZE_IN_WARP
#undef SET_SIZE_IN_THD
#undef SET_SIZE_IN_BITS

#undef SET_SIZE_X_IN_WARP
#undef SET_SIZE_Y_IN_WARP

#undef SET_SIZE_X_IN_BITS
#undef SET_SIZE_Y_IN_BITS

#undef CTA_SIZE_IN_WARP
#undef CTA_SIZE_IN_THD
#undef CTA_SIZE_IN_BITS

////////////////////////////////////////
// tiling size macros
////////////////////////////////////////

#undef TILE_M_PER_THD
#undef TILE_N_PER_THD

/////////////////////
// tile m

#undef TILE_M_V1_PER_CTA
#undef TILE_M_V2_PER_CTA
#undef TILE_M_V4_PER_CTA
#undef TILE_M_V8_PER_CTA

#undef TILE_M_V1_PER_WARP
#undef TILE_M_V2_PER_WARP
#undef TILE_M_V4_PER_WARP
#undef TILE_M_V8_PER_WARP

#undef TILE_M_V1_PER_THD
#undef TILE_M_V2_PER_THD
#undef TILE_M_V4_PER_THD
#undef TILE_M_V8_PER_THD

#undef TILE_M_V1_PER_MMA
#undef TILE_M_V2_PER_MMA
#undef TILE_M_V4_PER_MMA
#undef TILE_M_V8_PER_MMA

/////////////////////
// tile k

#undef TILE_K_V1_PER_CTA
#undef TILE_K_V2_PER_CTA
#undef TILE_K_V4_PER_CTA
#undef TILE_K_V8_PER_CTA

#undef TILE_K_V1_PER_WARP
#undef TILE_K_V2_PER_WARP
#undef TILE_K_V4_PER_WARP
#undef TILE_K_V8_PER_WARP

#undef TILE_K_V1_PER_THD
#undef TILE_K_V2_PER_THD
#undef TILE_K_V4_PER_THD
#undef TILE_K_V8_PER_THD

#undef TILE_K_V1_PER_KMA
#undef TILE_K_V2_PER_KMA
#undef TILE_K_V4_PER_KMA
#undef TILE_K_V8_PER_KMA


/////////////////////
// tile n

#undef TILE_N_V1_PER_CTA
#undef TILE_N_V2_PER_CTA
#undef TILE_N_V4_PER_CTA
#undef TILE_N_V8_PER_CTA

#undef TILE_N_V1_PER_WARP
#undef TILE_N_V2_PER_WARP
#undef TILE_N_V4_PER_WARP
#undef TILE_N_V8_PER_WARP

#undef TILE_N_V1_PER_THD
#undef TILE_N_V2_PER_THD
#undef TILE_N_V4_PER_THD
#undef TILE_N_V8_PER_THD

#undef TILE_N_V1_PER_MMA
#undef TILE_N_V2_PER_MMA
#undef TILE_N_V4_PER_MMA
#undef TILE_N_V8_PER_MMA


////////////////////////////////////////
// shared memory size macros
////////////////////////////////////////

#undef OUTPUT_STEPS

#undef N_ROWS_PER_SMEM_ROW
#undef K_ROWS_PER_SMEM_ROW

#undef OUTPUT_SIZE_X_IN_THD
#undef OUTPUT_SIZE_Y_IN_THD

////////////////////////////////////////
// main loop macros
////////////////////////////////////////

#undef C_ITEMS_PER_THD

////////////////////////////////////////
// load A and B from device memory macros
////////////////////////////////////////

#undef REG_dAv4_SIZE

#undef REG_dBv1_SIZE
#undef REG_dBv2_SIZE
#undef REG_dBv4_SIZE

#undef READ_dBv1_STEPS
#undef READ_dBv4_STEPS

#undef SET_dBv1_BOUND
#undef SET_dBv4_BOUND

////////////////////////////////////////
// shared memory size macros
////////////////////////////////////////

#undef USE_1BUF
#undef USE_2BUF

#undef SM_A_SIZE
#undef SM_B_SIZE
#undef SM_C_SIZE

#undef SM_A_1BUF
#undef SM_B_1BUF
#undef SM_C_1BUF

#undef SM_A_2BUF
#undef SM_B_2BUF
#undef SM_C_2BUF

#undef SM_A_V1_1BUF
#undef SM_B_V1_1BUF
#undef SM_C_V1_1BUF

#undef SM_A_V2_1BUF
#undef SM_B_V2_1BUF
#undef SM_C_V2_1BUF

#undef SM_A_V4_1BUF
#undef SM_B_V4_1BUF
#undef SM_C_V4_1BUF

#undef SM_A_V1_2BUF
#undef SM_B_V1_2BUF
#undef SM_C_V1_2BUF

#undef SM_A_V2_2BUF
#undef SM_B_V2_2BUF
#undef SM_C_V2_2BUF

#undef SM_A_V4_2BUF
#undef SM_B_V4_2BUF
#undef SM_C_V4_2BUF

#undef SM_BASE_V4_1BUF
#undef SM_BASE_V4_2BUF

#undef CVT_SM_PTR

#undef FWD_LUT

#undef FWD_FLT
#undef FWD_FLT1
#undef FLT_SIZE1
#undef FWD_FLT3
#undef FLT_SIZE3
#undef FWD_FLTN
#undef FLT_SIZEN

#undef FWD_FLT_SIZE1
#undef FWD_FLT_SIZE2
#undef FWD_FLT_SIZE4
#undef FWD_FLT_SIZE8
#undef FWD_FLT_SIZE16

////////////////////////////////////////
// mma macros
////////////////////////////////////////

#undef MMA_INST_OPCODE
#undef MMA_INST

#undef MMA_INST_ASCEND1
#undef MMA_INST_ASCEND2
#undef MMA_INST_ASCEND4
#undef MMA_INST_ASCEND8

#undef MMA_INST_DESCEND1
#undef MMA_INST_DESCEND2
#undef MMA_INST_DESCEND4
#undef MMA_INST_DESCEND8

#undef MMA_INST_1x1
#undef MMA_INST_1x2
#undef MMA_INST_1x4
#undef MMA_INST_1x8

#undef MMA_INST_2x1
#undef MMA_INST_2x2
#undef MMA_INST_2x4
#undef MMA_INST_2x8

#undef MMA_INST_4x1
#undef MMA_INST_4x2
#undef MMA_INST_4x4
#undef MMA_INST_4x8

#undef MMA_INST_8x1
#undef MMA_INST_8x2
#undef MMA_INST_8x4

#undef MMA_INSTS

/////////////////////////////////////////////////////
// reduce half2 macros
/////////////////////////////////////////////////////

#undef REDUCE_HALF2_SIZE4

#undef REDUCE_HALF2_1x4
#undef REDUCE_HALF2_3x4

#undef REDUCE

/////////////////////////////////////////////////////
// read sRv4 macros
/////////////////////////////////////////////////////

#undef READ_sRv4_SIZE1
#undef READ_sRv4_SIZE2
#undef READ_sRv4_SIZE4

#undef READ_sRv4

/////////////////////////////////////////////////////
// write sRv1 macros
/////////////////////////////////////////////////////

#undef WRITE_sRv2_SIZE1
#undef WRITE_sRv2_SIZE2
#undef WRITE_sRv2_SIZE4
#undef WRITE_sRv2_SIZE8

#undef WRITE_sRv2_1x1
#undef WRITE_sRv2_2x1
#undef WRITE_sRv2_4x1
#undef WRITE_sRv2_8x1
#undef WRITE_sRv2_16x1

#undef WRITE_sRv2_1x2
#undef WRITE_sRv2_2x2
#undef WRITE_sRv2_4x2
#undef WRITE_sRv2_8x2
#undef WRITE_sRv2_16x2

#undef WRITE_sRv2_1x4
#undef WRITE_sRv2_2x4
#undef WRITE_sRv2_4x4
#undef WRITE_sRv2_8x4
#undef WRITE_sRv2_16x4

#undef WRITE_sRv2_1x8
#undef WRITE_sRv2_2x8
#undef WRITE_sRv2_4x8
#undef WRITE_sRv2_8x8

#undef WRITE_sRv2

/////////////////////////////////////////////////////
// common load global memory macros
/////////////////////////////////////////////////////

//////////////////////////
// load dA
//////////////////////////

#undef LOAD_dAv4_SIZE_16TH
#undef LOAD_dAv4_SIZE_8TH
#undef LOAD_dAv4_SIZE_QTR
#undef LOAD_dAv4_SIZE_HALF
#undef LOAD_dAv4_SIZE1
#undef LOAD_dAv4_SIZE2
#undef LOAD_dAv4_SIZE4
#undef LOAD_dAv4_SIZE8
#undef LOAD_dAv4_SIZE16

#undef LOAD_dAv4

#undef SET_dAv4_BOUND 

//////////////////////////
// load dB
//////////////////////////

#undef LOAD_dBv4_SIZE_16TH
#undef LOAD_dBv4_SIZE_8TH
#undef LOAD_dBv4_SIZE_QTR
#undef LOAD_dBv4_SIZE_HALF
#undef LOAD_dBv4_SIZE1
#undef LOAD_dBv4_SIZE2
#undef LOAD_dBv4_SIZE4
#undef LOAD_dBv4_SIZE8
#undef LOAD_dBv4_SIZE16

#undef LOAD_dBv4

#undef SET_dBv4_BOUND 

/////////////////////////////////////////////////////
// common write shared memory macros
/////////////////////////////////////////////////////

#undef SWITCH_BUFFER

#undef FWD_KGROUP_ODD
#undef FWD_KGROUP_EVEN

#undef FWD_KGROUP_STEP1
#undef FWD_KGROUP_STEP2
#undef FWD_KGROUP_STEP3
#undef FWD_KGROUP_STEP4

#undef C_ITEMS_PER_THD
#undef HC_ITEMS_PER_THD
#undef Cv4_ITEMS_PER_THD

//////////////////////////
// write sA & sB
//////////////////////////

#undef WRITE_sUv4_SIZE_16TH
#undef WRITE_sUv4_SIZE_8TH
#undef WRITE_sUv4_SIZE_QTR
#undef WRITE_sUv4_SIZE_HALF
#undef WRITE_sUv4_SIZE1
#undef WRITE_sUv4_SIZE2
#undef WRITE_sUv4_SIZE4
#undef WRITE_sUv4_SIZE8
#undef WRITE_sUv4_SIZE16

#undef WRITE_sAv4
#undef WRITE_sBv4

/////////////////////////////////////////////////////
// read shared memory macros
/////////////////////////////////////////////////////

//////////////////////////
// read sA & sB
//////////////////////////

#undef REG_sAv1_SIZE
#undef REG_sBv1_SIZE

#undef READ_sUv1_SIZE1
#undef READ_sUv1_SIZE2
#undef READ_sUv1_SIZE4

#undef READ_sUv1_1x1
#undef READ_sUv1_2x1

#undef READ_sUv1_1x2
#undef READ_sUv1_2x2

#undef READ_sUv1_1x4
#undef READ_sUv1_2x4

#undef READ_sAv1
#undef READ_sBv1

/////////////////////////////////////////////////////
// precision half output
/////////////////////////////////////////////////////

#undef OUTPUT_PRC_HALF

#undef ADD_BIAS_V4

#undef FUSE_RELU_V4
#undef FUSE_CLIP_V4
#undef FUSE_PRELU_V4
#undef FUSE_ELT_V4

#undef SET_CONCAT_OFF_V4


#undef cvtOutData
#undef quantOutData
#undef deQuantData
#undef LOAD_SCALE
#undef packchar4
#undef TILE_N_IN_MMA_PER_WARP
#undef MMAs_PER_REDUCE_ROW
#undef SWIZZLE_GROUP
#undef MAX
#undef MIN
