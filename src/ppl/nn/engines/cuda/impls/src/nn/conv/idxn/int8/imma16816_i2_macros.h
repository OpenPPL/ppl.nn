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
// imma macros
////////////////////////////////////////

// int8 input, int32 output
#define MMA_INST_OPCODE \
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"

// operand c is omitted
#define MMA_INST(_d0, _d1, _d2, _d3, _a0, _a1, _b) \
        asm volatile(MMA_INST_OPCODE:   "=r"(_d0), "=r"(_d1), "=r"(_d2), "=r"(_d3): "r"(_a0), "r"(_a1), "r"(_b), "r"(_d0), "r"(_d1), "r"(_d2), "r"(_d3));

#define MMA_INST_2INT_ASCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _C[_C_off + _C_stride],     _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off]); \
        }
        
#define MMA_INST_2INT_ASCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _C[_C_off + _C_stride],     _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _C[_C_off + _C_stride + 2], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \
        }
        
#define MMA_INST_2INT_ASCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _C[_C_off + _C_stride],     _C[_C_off + _C_stride + 1], _a0, _a1, _Bv1[_Bv1_off]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _C[_C_off + _C_stride + 2], _C[_C_off + _C_stride + 3], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off + 4], _C[_C_off + 5], _C[_C_off + _C_stride + 4], _C[_C_off + _C_stride + 5], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off + 6], _C[_C_off + 7], _C[_C_off + _C_stride + 6], _C[_C_off + _C_stride + 7], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \
        }
        
#define MMA_INST_2INT_ASCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],      _C[_C_off + 1],  _C[_C_off + _C_stride],      _C[_C_off + _C_stride + 1],  _a0, _a1, _Bv1[_Bv1_off]); \
            MMA_INST(_C[_C_off + 2],  _C[_C_off + 3],  _C[_C_off + _C_stride + 2],  _C[_C_off + _C_stride + 3],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off + 4],  _C[_C_off + 5],  _C[_C_off + _C_stride + 4],  _C[_C_off + _C_stride + 5],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off + 6],  _C[_C_off + 7],  _C[_C_off + _C_stride + 6],  _C[_C_off + _C_stride + 7],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off + 8],  _C[_C_off + 9],  _C[_C_off + _C_stride + 8],  _C[_C_off + _C_stride + 9],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 4]); \
            MMA_INST(_C[_C_off + 10], _C[_C_off + 11], _C[_C_off + _C_stride + 10], _C[_C_off + _C_stride + 11], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 5]); \
            MMA_INST(_C[_C_off + 12], _C[_C_off + 13], _C[_C_off + _C_stride + 12], _C[_C_off + _C_stride + 13], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 6]); \
            MMA_INST(_C[_C_off + 14], _C[_C_off + 15], _C[_C_off + _C_stride + 14], _C[_C_off + _C_stride + 15], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 7]); \
        }
        
#define MMA_INST_2INT_DESCEND1(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - _C_stride - 1],  _C[_C_off - _C_stride],     _C[_C_off - 1],  _C[_C_off],     _a0, _a1, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_DESCEND2(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - _C_stride - 1],  _C[_C_off - _C_stride],     _C[_C_off - 1],  _C[_C_off],     _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off - _C_stride - 3],  _C[_C_off - _C_stride - 2], _C[_C_off - 3],  _C[_C_off - 2], _a0, _a1, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_DESCEND4(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - _C_stride - 1],  _C[_C_off - _C_stride],     _C[_C_off - 1],  _C[_C_off],     _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off - _C_stride - 3],  _C[_C_off - _C_stride - 2], _C[_C_off - 3],  _C[_C_off - 2], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off - _C_stride - 5],  _C[_C_off - _C_stride - 4], _C[_C_off - 5],  _C[_C_off - 4], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off - _C_stride - 7],  _C[_C_off - _C_stride - 6], _C[_C_off - 7],  _C[_C_off - 6], _a0, _a1, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_DESCEND8(_C, _C_off, _C_stride, _a0, _a1, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - _C_stride - 1],  _C[_C_off - _C_stride],      _C[_C_off - 1],  _C[_C_off],      _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 7]); \
            MMA_INST(_C[_C_off - _C_stride - 3],  _C[_C_off - _C_stride - 2],  _C[_C_off - 3],  _C[_C_off - 2],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 6]); \
            MMA_INST(_C[_C_off - _C_stride - 5],  _C[_C_off - _C_stride - 4],  _C[_C_off - 5],  _C[_C_off - 4],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 5]); \
            MMA_INST(_C[_C_off - _C_stride - 7],  _C[_C_off - _C_stride - 6],  _C[_C_off - 7],  _C[_C_off - 6],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 4]); \
            MMA_INST(_C[_C_off - _C_stride - 9],  _C[_C_off - _C_stride - 8],  _C[_C_off - 9],  _C[_C_off - 8],  _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off - _C_stride - 11], _C[_C_off - _C_stride - 10], _C[_C_off - 11], _C[_C_off - 10], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off - _C_stride - 13], _C[_C_off - _C_stride - 12], _C[_C_off - 13], _C[_C_off - 12], _a0, _a1, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off - _C_stride - 15], _C[_C_off - _C_stride - 14], _C[_C_off - 15], _C[_C_off - 14], _a0, _a1, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_1x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \
        }

#define MMA_INST_2INT_1x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \
        }

#define MMA_INST_2INT_1x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \
        }

#define MMA_INST_2INT_1x8(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0], _Av1[0 + _2INT_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1], _Av1[1 + _2INT_], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 7,  TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 7,  TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 15, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 15, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 31, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 31, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x8(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],             _Av1[0 + _2INT_],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 63, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_], _Av1[0 + _2INT_ + _2INT_X2_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],             _Av1[1 + _2INT_],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 63, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_], _Av1[1 + _2INT_ + _2INT_X2_], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 7,  TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 15, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 7,  TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 15, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 15, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 16, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 31, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 15, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 16, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 31, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 31, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 32, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 63, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 31, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 32, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 63, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x8(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 63, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND8 (_C, 64, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 127,TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND8 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 63, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND8 (_C, 64, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 127,TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_8x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 7,  TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 15, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 16, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 4], _Av1[0 + _2INT_ + _2INT_X2_ * 4], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 23, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 5], _Av1[0 + _2INT_ + _2INT_X2_ * 5], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 24, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 6], _Av1[0 + _2INT_ + _2INT_X2_ * 6], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 31, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 7], _Av1[0 + _2INT_ + _2INT_X2_ * 7], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 7,  TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 15, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 16, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 4], _Av1[1 + _2INT_ + _2INT_X2_ * 4], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 23, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 5], _Av1[1 + _2INT_ + _2INT_X2_ * 5], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 24, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 6], _Av1[1 + _2INT_ + _2INT_X2_ * 6], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 31, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 7], _Av1[1 + _2INT_ + _2INT_X2_ * 7], _Bv1, 1); \
        }

#define MMA_INST_2INT_8x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 15, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 16, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 31, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 32, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 4], _Av1[0 + _2INT_ + _2INT_X2_ * 4], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 47, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 5], _Av1[0 + _2INT_ + _2INT_X2_ * 5], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 48, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 6], _Av1[0 + _2INT_ + _2INT_X2_ * 6], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 63, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 7], _Av1[0 + _2INT_ + _2INT_X2_ * 7], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 15, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 16, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 31, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 32, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 4], _Av1[1 + _2INT_ + _2INT_X2_ * 4], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 47, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 5], _Av1[1 + _2INT_ + _2INT_X2_ * 5], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 48, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 6], _Av1[1 + _2INT_ + _2INT_X2_ * 6], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 63, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 7], _Av1[1 + _2INT_ + _2INT_X2_ * 7], _Bv1, 1); \
        }

#define MMA_INST_2INT_8x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[0],                 _Av1[0 + _2INT_],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 31, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 1], _Av1[0 + _2INT_ + _2INT_X2_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 32, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 2], _Av1[0 + _2INT_ + _2INT_X2_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 63, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 3], _Av1[0 + _2INT_ + _2INT_X2_ * 3], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 64, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 4], _Av1[0 + _2INT_ + _2INT_X2_ * 4], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 95, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 5], _Av1[0 + _2INT_ + _2INT_X2_ * 5], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 96, TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 6], _Av1[0 + _2INT_ + _2INT_X2_ * 6], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 127,TILE_N_V1_PER_THD, _Av1[0 + _2INT_X2_ * 7], _Av1[0 + _2INT_ + _2INT_X2_ * 7], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  TILE_N_V1_PER_THD, _Av1[1],                 _Av1[1 + _2INT_],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 31, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 1], _Av1[1 + _2INT_ + _2INT_X2_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 32, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 2], _Av1[1 + _2INT_ + _2INT_X2_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 63, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 3], _Av1[1 + _2INT_ + _2INT_X2_ * 3], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 64, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 4], _Av1[1 + _2INT_ + _2INT_X2_ * 4], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 95, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 5], _Av1[1 + _2INT_ + _2INT_X2_ * 5], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 96, TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 6], _Av1[1 + _2INT_ + _2INT_X2_ * 6], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 127,TILE_N_V1_PER_THD, _Av1[1 + _2INT_X2_ * 7], _Av1[1 + _2INT_ + _2INT_X2_ * 7], _Bv1, 1); \
        }

