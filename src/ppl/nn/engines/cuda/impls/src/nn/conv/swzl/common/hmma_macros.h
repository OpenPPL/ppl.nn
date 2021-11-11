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
// hmma macros
////////////////////////////////////////

#define MMA_INST_OPCODE \
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
        
#define MMA_INST(_d0, _d1, _b0, _b1, _a) \
        asm volatile(MMA_INST_OPCODE:   "=r"(_d0),   "=r"(_d1): "r"(_b0), "r"(_b1), "r"(_a),  "r"(_d0),   "r"(_d1));

#define MMA_INST_ASCEND1(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[0]); \
        }
        
#define MMA_INST_ASCEND2(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[0]); \
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _b0, _b1, _A[1]); \
        }
        
#define MMA_INST_ASCEND4(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[0]); \
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _b0, _b1, _A[1]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _b0, _b1, _A[2]); \
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _b0, _b1, _A[3]); \
        }
        
#define MMA_INST_ASCEND8(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[0]); \
            MMA_INST(_C[_C_off + 1], _C[_C_off + _C_stride + 1], _b0, _b1, _A[1]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + _C_stride + 2], _b0, _b1, _A[2]); \
            MMA_INST(_C[_C_off + 3], _C[_C_off + _C_stride + 3], _b0, _b1, _A[3]); \
            MMA_INST(_C[_C_off + 4], _C[_C_off + _C_stride + 4], _b0, _b1, _A[4]); \
            MMA_INST(_C[_C_off + 5], _C[_C_off + _C_stride + 5], _b0, _b1, _A[5]); \
            MMA_INST(_C[_C_off + 6], _C[_C_off + _C_stride + 6], _b0, _b1, _A[6]); \
            MMA_INST(_C[_C_off + 7], _C[_C_off + _C_stride + 7], _b0, _b1, _A[7]); \
        }
        
       
#define MMA_INST_DESCEND1(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[0]); \
        }

#define MMA_INST_DESCEND2(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[1]); \
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _b0, _b1, _A[0]); \
        }

#define MMA_INST_DESCEND4(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[3]); \
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _b0, _b1, _A[2]); \
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _b0, _b1, _A[1]); \
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _b0, _b1, _A[0]); \
        }

#define MMA_INST_DESCEND8(_C, _C_off, _C_stride, _b0, _b1, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + _C_stride],     _b0, _b1, _A[7]); \
            MMA_INST(_C[_C_off - 1], _C[_C_off + _C_stride - 1], _b0, _b1, _A[6]); \
            MMA_INST(_C[_C_off - 2], _C[_C_off + _C_stride - 2], _b0, _b1, _A[5]); \
            MMA_INST(_C[_C_off - 3], _C[_C_off + _C_stride - 3], _b0, _b1, _A[4]); \
            MMA_INST(_C[_C_off - 4], _C[_C_off + _C_stride - 4], _b0, _b1, _A[3]); \
            MMA_INST(_C[_C_off - 5], _C[_C_off + _C_stride - 5], _b0, _b1, _A[2]); \
            MMA_INST(_C[_C_off - 6], _C[_C_off + _C_stride - 6], _b0, _b1, _A[1]); \
            MMA_INST(_C[_C_off - 7], _C[_C_off + _C_stride - 7], _b0, _b1, _A[0]); \
        }

#define MMA_INST_1x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
        }

#define MMA_INST_2x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
        }

#define MMA_INST_4x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
        }

#define MMA_INST_8x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND8  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
        }

#define MMA_INST_1x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND1 (_C, 2,  TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
        }

#define MMA_INST_2x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND2 (_C, 5,  TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
        }

#define MMA_INST_4x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND4 (_C, 11, TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
        }

#define MMA_INST_8x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND8  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND8 (_C, 23, TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
        }

#define MMA_INST_1x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND1 (_C, 2,  TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
            \
            MMA_INST_ASCEND1  (_C, 4,  TILE_M_V2_PER_THD, _B[4], _B[5], _A); \
            MMA_INST_DESCEND1 (_C, 6,  TILE_M_V2_PER_THD, _B[6], _B[7], _A); \
        }

#define MMA_INST_2x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND2 (_C, 5,  TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
            \
            MMA_INST_ASCEND2  (_C, 8,  TILE_M_V2_PER_THD, _B[4], _B[5], _A); \
            MMA_INST_DESCEND2 (_C, 13, TILE_M_V2_PER_THD, _B[6], _B[7], _A); \
        }

#define MMA_INST_4x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND4 (_C, 11, TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
            \
            MMA_INST_ASCEND4  (_C, 16, TILE_M_V2_PER_THD, _B[4], _B[5], _A); \
            MMA_INST_DESCEND4 (_C, 27, TILE_M_V2_PER_THD, _B[6], _B[7], _A); \
        }

#define MMA_INST_8x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND8  (_C, 0,  TILE_M_V2_PER_THD, _B[0], _B[1], _A); \
            MMA_INST_DESCEND8 (_C, 23, TILE_M_V2_PER_THD, _B[2], _B[3], _A); \
            \
            MMA_INST_ASCEND8  (_C, 32, TILE_M_V2_PER_THD, _B[4], _B[5], _A); \
            MMA_INST_DESCEND8 (_C, 55, TILE_M_V2_PER_THD, _B[6], _B[7], _A); \
        }

#define MMA_INST_1x8(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  TILE_M_V2_PER_THD, _B[0],  _B[1],  _A); \
            MMA_INST_DESCEND1 (_C, 2,  TILE_M_V2_PER_THD, _B[2],  _B[3],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 4,  TILE_M_V2_PER_THD, _B[4],  _B[5],  _A); \
            MMA_INST_DESCEND1 (_C, 6,  TILE_M_V2_PER_THD, _B[6],  _B[7],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 8,  TILE_M_V2_PER_THD, _B[8],  _B[9],  _A); \
            MMA_INST_DESCEND1 (_C, 10, TILE_M_V2_PER_THD, _B[10], _B[11], _A); \
            \
            MMA_INST_ASCEND1  (_C, 12, TILE_M_V2_PER_THD, _B[12], _B[13], _A); \
            MMA_INST_DESCEND1 (_C, 14, TILE_M_V2_PER_THD, _B[14], _B[15], _A); \
        }

#define MMA_INST_2x8(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  TILE_M_V2_PER_THD, _B[0],  _B[1],  _A); \
            MMA_INST_DESCEND2 (_C, 5,  TILE_M_V2_PER_THD, _B[2],  _B[3],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 8,  TILE_M_V2_PER_THD, _B[4],  _B[5],  _A); \
            MMA_INST_DESCEND2 (_C, 13, TILE_M_V2_PER_THD, _B[6],  _B[7],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 16, TILE_M_V2_PER_THD, _B[8],  _B[9],  _A); \
            MMA_INST_DESCEND2 (_C, 21, TILE_M_V2_PER_THD, _B[10], _B[11], _A); \
            \
            MMA_INST_ASCEND2  (_C, 24, TILE_M_V2_PER_THD, _B[12], _B[13], _A); \
            MMA_INST_DESCEND2 (_C, 29, TILE_M_V2_PER_THD, _B[14], _B[15], _A); \
        }

#define MMA_INST_4x8(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  TILE_M_V2_PER_THD, _B[0],  _B[1],  _A); \
            MMA_INST_DESCEND4 (_C, 11, TILE_M_V2_PER_THD, _B[2],  _B[3],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 16, TILE_M_V2_PER_THD, _B[4],  _B[5],  _A); \
            MMA_INST_DESCEND4 (_C, 27, TILE_M_V2_PER_THD, _B[6],  _B[7],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 32, TILE_M_V2_PER_THD, _B[8],  _B[9],  _A); \
            MMA_INST_DESCEND4 (_C, 43, TILE_M_V2_PER_THD, _B[10], _B[11], _A); \
            \
            MMA_INST_ASCEND4  (_C, 48, TILE_M_V2_PER_THD, _B[12], _B[13], _A); \
            MMA_INST_DESCEND4 (_C, 59, TILE_M_V2_PER_THD, _B[14], _B[15], _A); \
        }

