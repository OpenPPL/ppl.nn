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
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"

// operand c is omitted
#define MMA_INST(_d0, _d1, _a, _b) \
        asm volatile(MMA_INST_OPCODE:   "=r"(_d0),   "=r"(_d1): "r"(_a), "r"(_b),  "r"(_d0),   "r"(_d1));


#define MMA_INST_2INT_ASCEND1(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _a, _Bv1[_Bv1_off]); \
        }
        
#define MMA_INST_2INT_ASCEND2(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _a, _Bv1[_Bv1_off]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
        }
        
#define MMA_INST_2INT_ASCEND4(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _a, _Bv1[_Bv1_off]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off + 4], _C[_C_off + 5], _a, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off + 6], _C[_C_off + 7], _a, _Bv1[_Bv1_off + _2INT_ * 3]); \
        }
        
#define MMA_INST_2INT_ASCEND8(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],      _C[_C_off +  1], _a, _Bv1[_Bv1_off]); \
            MMA_INST(_C[_C_off +  2], _C[_C_off +  3], _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off +  4], _C[_C_off +  5], _a, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off +  6], _C[_C_off +  7], _a, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off +  8], _C[_C_off +  9], _a, _Bv1[_Bv1_off + _2INT_ * 4]); \
            MMA_INST(_C[_C_off + 10], _C[_C_off + 11], _a, _Bv1[_Bv1_off + _2INT_ * 5]); \
            MMA_INST(_C[_C_off + 12], _C[_C_off + 13], _a, _Bv1[_Bv1_off + _2INT_ * 6]); \
            MMA_INST(_C[_C_off + 14], _C[_C_off + 15], _a, _Bv1[_Bv1_off + _2INT_ * 7]); \
        }
        
#define MMA_INST_2INT_ASCEND16(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off],      _C[_C_off +  1], _a, _Bv1[_Bv1_off]); \
            MMA_INST(_C[_C_off +  2], _C[_C_off +  3], _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off +  4], _C[_C_off +  5], _a, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off +  6], _C[_C_off +  7], _a, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off +  8], _C[_C_off +  9], _a, _Bv1[_Bv1_off + _2INT_ * 4]); \
            MMA_INST(_C[_C_off + 10], _C[_C_off + 11], _a, _Bv1[_Bv1_off + _2INT_ * 5]); \
            MMA_INST(_C[_C_off + 12], _C[_C_off + 13], _a, _Bv1[_Bv1_off + _2INT_ * 6]); \
            MMA_INST(_C[_C_off + 14], _C[_C_off + 15], _a, _Bv1[_Bv1_off + _2INT_ * 7]); \
            MMA_INST(_C[_C_off + 16], _C[_C_off + 17], _a, _Bv1[_Bv1_off + _2INT_ * 8]); \
            MMA_INST(_C[_C_off + 18], _C[_C_off + 19], _a, _Bv1[_Bv1_off + _2INT_ * 9]); \
            MMA_INST(_C[_C_off + 20], _C[_C_off + 21], _a, _Bv1[_Bv1_off + _2INT_ * 10]); \
            MMA_INST(_C[_C_off + 22], _C[_C_off + 23], _a, _Bv1[_Bv1_off + _2INT_ * 11]); \
            MMA_INST(_C[_C_off + 24], _C[_C_off + 25], _a, _Bv1[_Bv1_off + _2INT_ * 12]); \
            MMA_INST(_C[_C_off + 26], _C[_C_off + 27], _a, _Bv1[_Bv1_off + _2INT_ * 13]); \
            MMA_INST(_C[_C_off + 28], _C[_C_off + 29], _a, _Bv1[_Bv1_off + _2INT_ * 14]); \
            MMA_INST(_C[_C_off + 30], _C[_C_off + 31], _a, _Bv1[_Bv1_off + _2INT_ * 15]); \
        }

#define MMA_INST_2INT_DESCEND1(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - 1], _C[_C_off],     _a, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_DESCEND2(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - 1], _C[_C_off],     _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off - 3], _C[_C_off - 2], _a, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_DESCEND4(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - 1], _C[_C_off],     _a, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off - 3], _C[_C_off - 2], _a, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off - 5], _C[_C_off - 4], _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off - 7], _C[_C_off - 6], _a, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_DESCEND8(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off -  1], _C[_C_off],      _a, _Bv1[_Bv1_off + _2INT_ * 7]); \
            MMA_INST(_C[_C_off -  3], _C[_C_off -  2], _a, _Bv1[_Bv1_off + _2INT_ * 6]); \
            MMA_INST(_C[_C_off -  5], _C[_C_off -  4], _a, _Bv1[_Bv1_off + _2INT_ * 5]); \
            MMA_INST(_C[_C_off -  7], _C[_C_off -  6], _a, _Bv1[_Bv1_off + _2INT_ * 4]); \
            MMA_INST(_C[_C_off -  9], _C[_C_off -  8], _a, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off - 11], _C[_C_off - 10], _a, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off - 13], _C[_C_off - 12], _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off - 15], _C[_C_off - 14], _a, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_DESCEND16(_C, _C_off, _a, _Bv1, _Bv1_off) \
        { \
            MMA_INST(_C[_C_off - 1],  _C[_C_off],      _a, _Bv1[_Bv1_off + _2INT_ * 15]); \
            MMA_INST(_C[_C_off - 3],  _C[_C_off - 2],  _a, _Bv1[_Bv1_off + _2INT_ * 14]); \
            MMA_INST(_C[_C_off - 5],  _C[_C_off - 4],  _a, _Bv1[_Bv1_off + _2INT_ * 13]); \
            MMA_INST(_C[_C_off - 7],  _C[_C_off - 6],  _a, _Bv1[_Bv1_off + _2INT_ * 12]); \
            MMA_INST(_C[_C_off - 9],  _C[_C_off - 8],  _a, _Bv1[_Bv1_off + _2INT_ * 11]); \
            MMA_INST(_C[_C_off - 11], _C[_C_off - 10], _a, _Bv1[_Bv1_off + _2INT_ * 10]); \
            MMA_INST(_C[_C_off - 13], _C[_C_off - 12], _a, _Bv1[_Bv1_off + _2INT_ * 9]); \
            MMA_INST(_C[_C_off - 15], _C[_C_off - 14], _a, _Bv1[_Bv1_off + _2INT_ * 8]); \
            MMA_INST(_C[_C_off - 17], _C[_C_off - 16], _a, _Bv1[_Bv1_off + _2INT_ * 7]); \
            MMA_INST(_C[_C_off - 19], _C[_C_off - 18], _a, _Bv1[_Bv1_off + _2INT_ * 6]); \
            MMA_INST(_C[_C_off - 21], _C[_C_off - 20], _a, _Bv1[_Bv1_off + _2INT_ * 5]); \
            MMA_INST(_C[_C_off - 23], _C[_C_off - 22], _a, _Bv1[_Bv1_off + _2INT_ * 4]); \
            MMA_INST(_C[_C_off - 25], _C[_C_off - 24], _a, _Bv1[_Bv1_off + _2INT_ * 3]); \
            MMA_INST(_C[_C_off - 27], _C[_C_off - 26], _a, _Bv1[_Bv1_off + _2INT_ * 2]); \
            MMA_INST(_C[_C_off - 29], _C[_C_off - 28], _a, _Bv1[_Bv1_off + _2INT_ * 1]); \
            MMA_INST(_C[_C_off - 31], _C[_C_off - 30], _a, _Bv1[_Bv1_off]); \
        }

#define MMA_INST_2INT_1x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[0], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[1], _Bv1, 1); \
        }

#define MMA_INST_2INT_1x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[0], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[1], _Bv1, 1); \
        }

#define MMA_INST_2INT_1x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[0], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[1], _Bv1, 1); \
        }

#define MMA_INST_2INT_1x8(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND8 (_C, 0,  _Av1[0], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND8 (_C, 0,  _Av1[1], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[0],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[0 + _2INT_X1_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[1],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[1 + _2INT_X1_], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[0],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[0 + _2INT_X1_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[1],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[1 + _2INT_X1_], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[0],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 15, _Av1[0 + _2INT_X1_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[1],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 15, _Av1[1 + _2INT_X1_], _Bv1, 1); \
        }

#define MMA_INST_2INT_2x8(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND8 (_C, 0,  _Av1[0],             _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 31, _Av1[0 + _2INT_X1_], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND8 (_C, 0,  _Av1[1],             _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 31, _Av1[1 + _2INT_X1_], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 4,  _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 7,  _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 4,  _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 7,  _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 8,  _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 15, _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 8,  _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 15, _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 15, _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 16, _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 31, _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 15, _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 16, _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 31, _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_4x8(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND8 (_C, 0,  _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 31, _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND8 (_C, 32, _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 63, _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND8 (_C, 0,  _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 31, _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND8 (_C, 32, _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 63, _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
        }

#define MMA_INST_2INT_8x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 4,  _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 7,  _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  _Av1[0 + _2INT_X1_ * 4], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 11, _Av1[0 + _2INT_X1_ * 5], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 12, _Av1[0 + _2INT_X1_ * 6], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 15, _Av1[0 + _2INT_X1_ * 7], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 4,  _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 7,  _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  _Av1[1 + _2INT_X1_ * 4], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 11, _Av1[1 + _2INT_X1_ * 5], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 12, _Av1[1 + _2INT_X1_ * 6], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 15, _Av1[1 + _2INT_X1_ * 7], _Bv1, 1); \
        }

#define MMA_INST_2INT_8x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 8,  _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 15, _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 16, _Av1[0 + _2INT_X1_ * 4], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 23, _Av1[0 + _2INT_X1_ * 5], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 24, _Av1[0 + _2INT_X1_ * 6], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 31, _Av1[0 + _2INT_X1_ * 7], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 8,  _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 15, _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 16, _Av1[1 + _2INT_X1_ * 4], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 23, _Av1[1 + _2INT_X1_ * 5], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 24, _Av1[1 + _2INT_X1_ * 6], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 31, _Av1[1 + _2INT_X1_ * 7], _Bv1, 1); \
        }

#define MMA_INST_2INT_8x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 15, _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 16, _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 31, _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 32, _Av1[0 + _2INT_X1_ * 4], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 47, _Av1[0 + _2INT_X1_ * 5], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 48, _Av1[0 + _2INT_X1_ * 6], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 63, _Av1[0 + _2INT_X1_ * 7], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,  _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 15, _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 16, _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 31, _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 32, _Av1[1 + _2INT_X1_ * 4], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 47, _Av1[1 + _2INT_X1_ * 5], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 48, _Av1[1 + _2INT_X1_ * 6], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 63, _Av1[1 + _2INT_X1_ * 7], _Bv1, 1); \
        }

#define MMA_INST_2INT_8x8(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND8 (_C, 0,   _Av1[0],                 _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 31,  _Av1[0 + _2INT_X1_ * 1], _Bv1, 0); \
            MMA_INST_2INT_ASCEND8 (_C, 32,  _Av1[0 + _2INT_X1_ * 2], _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 63,  _Av1[0 + _2INT_X1_ * 3], _Bv1, 0); \
            MMA_INST_2INT_ASCEND8 (_C, 64,  _Av1[0 + _2INT_X1_ * 4], _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 95,  _Av1[0 + _2INT_X1_ * 5], _Bv1, 0); \
            MMA_INST_2INT_ASCEND8 (_C, 96,  _Av1[0 + _2INT_X1_ * 6], _Bv1, 0); \
            MMA_INST_2INT_DESCEND8(_C, 127, _Av1[0 + _2INT_X1_ * 7], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND8 (_C, 0,   _Av1[1],                 _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 31,  _Av1[1 + _2INT_X1_ * 1], _Bv1, 1); \
            MMA_INST_2INT_ASCEND8 (_C, 32,  _Av1[1 + _2INT_X1_ * 2], _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 63,  _Av1[1 + _2INT_X1_ * 3], _Bv1, 1); \
            MMA_INST_2INT_ASCEND8 (_C, 64,  _Av1[1 + _2INT_X1_ * 4], _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 95,  _Av1[1 + _2INT_X1_ * 5], _Bv1, 1); \
            MMA_INST_2INT_ASCEND8 (_C, 96,  _Av1[1 + _2INT_X1_ * 6], _Bv1, 1); \
            MMA_INST_2INT_DESCEND8(_C, 127, _Av1[1 + _2INT_X1_ * 7], _Bv1, 1); \
        }

#define MMA_INST_2INT_16x1(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[0],                  _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[0 + _2INT_X1_ * 1],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 4,  _Av1[0 + _2INT_X1_ * 2],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 7,  _Av1[0 + _2INT_X1_ * 3],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  _Av1[0 + _2INT_X1_ * 4],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 11, _Av1[0 + _2INT_X1_ * 5],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 12, _Av1[0 + _2INT_X1_ * 6],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 15, _Av1[0 + _2INT_X1_ * 7],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 16, _Av1[0 + _2INT_X1_ * 8],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 19, _Av1[0 + _2INT_X1_ * 9],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 20, _Av1[0 + _2INT_X1_ * 10], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 23, _Av1[0 + _2INT_X1_ * 11], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 24, _Av1[0 + _2INT_X1_ * 12], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 27, _Av1[0 + _2INT_X1_ * 13], _Bv1, 0); \
            MMA_INST_2INT_ASCEND1 (_C, 28, _Av1[0 + _2INT_X1_ * 14], _Bv1, 0); \
            MMA_INST_2INT_DESCEND1(_C, 31, _Av1[0 + _2INT_X1_ * 15], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND1 (_C, 0,  _Av1[1],                  _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 3,  _Av1[1 + _2INT_X1_ * 1],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 4,  _Av1[1 + _2INT_X1_ * 2],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 7,  _Av1[1 + _2INT_X1_ * 3],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 8,  _Av1[1 + _2INT_X1_ * 4],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 11, _Av1[1 + _2INT_X1_ * 5],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 12, _Av1[1 + _2INT_X1_ * 6],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 15, _Av1[1 + _2INT_X1_ * 7],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 16, _Av1[1 + _2INT_X1_ * 8],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 19, _Av1[1 + _2INT_X1_ * 9],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 20, _Av1[1 + _2INT_X1_ * 10], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 23, _Av1[1 + _2INT_X1_ * 11], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 24, _Av1[1 + _2INT_X1_ * 12], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 27, _Av1[1 + _2INT_X1_ * 13], _Bv1, 1); \
            MMA_INST_2INT_ASCEND1 (_C, 28, _Av1[1 + _2INT_X1_ * 14], _Bv1, 1); \
            MMA_INST_2INT_DESCEND1(_C, 31, _Av1[1 + _2INT_X1_ * 15], _Bv1, 1); \
        }

#define MMA_INST_2INT_16x2(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[0],                  _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[0 + _2INT_X1_ * 1],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 8,  _Av1[0 + _2INT_X1_ * 2],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 15, _Av1[0 + _2INT_X1_ * 3],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 16, _Av1[0 + _2INT_X1_ * 4],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 23, _Av1[0 + _2INT_X1_ * 5],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 24, _Av1[0 + _2INT_X1_ * 6],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 31, _Av1[0 + _2INT_X1_ * 7],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 32, _Av1[0 + _2INT_X1_ * 8],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 39, _Av1[0 + _2INT_X1_ * 9],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 40, _Av1[0 + _2INT_X1_ * 10], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 47, _Av1[0 + _2INT_X1_ * 11], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 48, _Av1[0 + _2INT_X1_ * 12], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 55, _Av1[0 + _2INT_X1_ * 13], _Bv1, 0); \
            MMA_INST_2INT_ASCEND2 (_C, 56, _Av1[0 + _2INT_X1_ * 14], _Bv1, 0); \
            MMA_INST_2INT_DESCEND2(_C, 63, _Av1[0 + _2INT_X1_ * 15], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND2 (_C, 0,  _Av1[1],                  _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 7,  _Av1[1 + _2INT_X1_ * 1],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 8,  _Av1[1 + _2INT_X1_ * 2],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 15, _Av1[1 + _2INT_X1_ * 3],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 16, _Av1[1 + _2INT_X1_ * 4],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 23, _Av1[1 + _2INT_X1_ * 5],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 24, _Av1[1 + _2INT_X1_ * 6],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 31, _Av1[1 + _2INT_X1_ * 7],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 32, _Av1[1 + _2INT_X1_ * 8],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 39, _Av1[1 + _2INT_X1_ * 9],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 40, _Av1[1 + _2INT_X1_ * 10], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 47, _Av1[1 + _2INT_X1_ * 11], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 48, _Av1[1 + _2INT_X1_ * 12], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 55, _Av1[1 + _2INT_X1_ * 13], _Bv1, 1); \
            MMA_INST_2INT_ASCEND2 (_C, 56, _Av1[1 + _2INT_X1_ * 14], _Bv1, 1); \
            MMA_INST_2INT_DESCEND2(_C, 63, _Av1[1 + _2INT_X1_ * 15], _Bv1, 1); \
        }

#define MMA_INST_2INT_16x4(_C, _Av1, _Bv1) \
        { \
            MMA_INST_2INT_ASCEND4 (_C, 0,   _Av1[0],                  _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 15,  _Av1[0 + _2INT_X1_ * 1],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 16,  _Av1[0 + _2INT_X1_ * 2],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 31,  _Av1[0 + _2INT_X1_ * 3],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 32,  _Av1[0 + _2INT_X1_ * 4],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 47,  _Av1[0 + _2INT_X1_ * 5],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 48,  _Av1[0 + _2INT_X1_ * 6],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 63,  _Av1[0 + _2INT_X1_ * 7],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 64,  _Av1[0 + _2INT_X1_ * 8],  _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 79,  _Av1[0 + _2INT_X1_ * 9],  _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 80,  _Av1[0 + _2INT_X1_ * 10], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 95,  _Av1[0 + _2INT_X1_ * 11], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 96,  _Av1[0 + _2INT_X1_ * 12], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 111, _Av1[0 + _2INT_X1_ * 13], _Bv1, 0); \
            MMA_INST_2INT_ASCEND4 (_C, 112, _Av1[0 + _2INT_X1_ * 14], _Bv1, 0); \
            MMA_INST_2INT_DESCEND4(_C, 127, _Av1[0 + _2INT_X1_ * 15], _Bv1, 0); \
            \
            MMA_INST_2INT_ASCEND4 (_C, 0,   _Av1[1],                  _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 15,  _Av1[1 + _2INT_X1_ * 1],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 16,  _Av1[1 + _2INT_X1_ * 2],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 31,  _Av1[1 + _2INT_X1_ * 3],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 32,  _Av1[1 + _2INT_X1_ * 4],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 47,  _Av1[1 + _2INT_X1_ * 5],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 48,  _Av1[1 + _2INT_X1_ * 6],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 63,  _Av1[1 + _2INT_X1_ * 7],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 64,  _Av1[1 + _2INT_X1_ * 8],  _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 79,  _Av1[1 + _2INT_X1_ * 9],  _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 80,  _Av1[1 + _2INT_X1_ * 10], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 95,  _Av1[1 + _2INT_X1_ * 11], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 96,  _Av1[1 + _2INT_X1_ * 12], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 111, _Av1[1 + _2INT_X1_ * 13], _Bv1, 1); \
            MMA_INST_2INT_ASCEND4 (_C, 112, _Av1[1 + _2INT_X1_ * 14], _Bv1, 1); \
            MMA_INST_2INT_DESCEND4(_C, 127, _Av1[1 + _2INT_X1_ * 15], _Bv1, 1); \
        }
