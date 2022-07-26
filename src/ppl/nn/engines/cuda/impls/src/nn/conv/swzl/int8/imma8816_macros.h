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

#define MMA_INST_ASCEND1(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _b0, _A[0]); \
        }
        
#define MMA_INST_ASCEND2(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _b0, _A[0]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _b0, _A[1]); \
        }
        
#define MMA_INST_ASCEND4(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off],     _C[_C_off + 1], _b0, _A[0]); \
            MMA_INST(_C[_C_off + 2], _C[_C_off + 3], _b0, _A[1]); \
            MMA_INST(_C[_C_off + 4], _C[_C_off + 5], _b0, _A[2]); \
            MMA_INST(_C[_C_off + 6], _C[_C_off + 7], _b0, _A[3]); \
        }
        
#define MMA_INST_ASCEND8(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off],      _C[_C_off + 1],  _b0, _A[0]); \
            MMA_INST(_C[_C_off + 2],  _C[_C_off + 3],  _b0, _A[1]); \
            MMA_INST(_C[_C_off + 4],  _C[_C_off + 5],  _b0, _A[2]); \
            MMA_INST(_C[_C_off + 6],  _C[_C_off + 7],  _b0, _A[3]); \
            MMA_INST(_C[_C_off + 8],  _C[_C_off + 9],  _b0, _A[4]); \
            MMA_INST(_C[_C_off + 10], _C[_C_off + 11], _b0, _A[5]); \
            MMA_INST(_C[_C_off + 12], _C[_C_off + 13], _b0, _A[6]); \
            MMA_INST(_C[_C_off + 14], _C[_C_off + 15], _b0, _A[7]); \
        }

#define MMA_INST_ASCEND16(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off],      _C[_C_off + 1],  _b0, _A[0]);  \
            MMA_INST(_C[_C_off + 2],  _C[_C_off + 3],  _b0, _A[1]);  \
            MMA_INST(_C[_C_off + 4],  _C[_C_off + 5],  _b0, _A[2]);  \
            MMA_INST(_C[_C_off + 6],  _C[_C_off + 7],  _b0, _A[3]);  \
            MMA_INST(_C[_C_off + 8],  _C[_C_off + 9],  _b0, _A[4]);  \
            MMA_INST(_C[_C_off + 10], _C[_C_off + 11], _b0, _A[5]);  \
            MMA_INST(_C[_C_off + 12], _C[_C_off + 13], _b0, _A[6]);  \
            MMA_INST(_C[_C_off + 14], _C[_C_off + 15], _b0, _A[7]);  \
            MMA_INST(_C[_C_off + 16], _C[_C_off + 17], _b0, _A[8]);  \
            MMA_INST(_C[_C_off + 18], _C[_C_off + 19], _b0, _A[9]);  \
            MMA_INST(_C[_C_off + 20], _C[_C_off + 21], _b0, _A[10]); \
            MMA_INST(_C[_C_off + 22], _C[_C_off + 23], _b0, _A[11]); \
            MMA_INST(_C[_C_off + 24], _C[_C_off + 25], _b0, _A[12]); \
            MMA_INST(_C[_C_off + 26], _C[_C_off + 27], _b0, _A[13]); \
            MMA_INST(_C[_C_off + 28], _C[_C_off + 29], _b0, _A[14]); \
            MMA_INST(_C[_C_off + 30], _C[_C_off + 31], _b0, _A[15]); \
        }

#define MMA_INST_DESCEND1(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off - 1], _C[_C_off - 0],   _b0, _A[0]); \
        }

#define MMA_INST_DESCEND2(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off - 1], _C[_C_off - 0],   _b0, _A[1]); \
            MMA_INST(_C[_C_off - 3], _C[_C_off - 2],   _b0, _A[0]); \
        }

#define MMA_INST_DESCEND4(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off - 1], _C[_C_off - 0],   _b0, _A[3]); \
            MMA_INST(_C[_C_off - 3], _C[_C_off - 2],   _b0, _A[2]); \
            MMA_INST(_C[_C_off - 5], _C[_C_off - 4],   _b0, _A[1]); \
            MMA_INST(_C[_C_off - 7], _C[_C_off - 6],   _b0, _A[0]); \
        }

#define MMA_INST_DESCEND8(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off - 1],  _C[_C_off - 0],  _b0, _A[7]); \
            MMA_INST(_C[_C_off - 3],  _C[_C_off - 2],  _b0, _A[6]); \
            MMA_INST(_C[_C_off - 5],  _C[_C_off - 4],  _b0, _A[5]); \
            MMA_INST(_C[_C_off - 7],  _C[_C_off - 6],  _b0, _A[4]); \
            MMA_INST(_C[_C_off - 9],  _C[_C_off - 8],  _b0, _A[3]); \
            MMA_INST(_C[_C_off - 11], _C[_C_off - 10], _b0, _A[2]); \
            MMA_INST(_C[_C_off - 13], _C[_C_off - 12], _b0, _A[1]); \
            MMA_INST(_C[_C_off - 15], _C[_C_off - 14], _b0, _A[0]); \
        }

#define MMA_INST_DESCEND16(_C, _C_off, _b0, _A) \
        { \
            MMA_INST(_C[_C_off - 1],  _C[_C_off - 0],  _b0, _A[15]); \
            MMA_INST(_C[_C_off - 3],  _C[_C_off - 2],  _b0, _A[14]); \
            MMA_INST(_C[_C_off - 5],  _C[_C_off - 4],  _b0, _A[13]); \
            MMA_INST(_C[_C_off - 7],  _C[_C_off - 6],  _b0, _A[12]); \
            MMA_INST(_C[_C_off - 9],  _C[_C_off - 8],  _b0, _A[11]); \
            MMA_INST(_C[_C_off - 11], _C[_C_off - 10], _b0, _A[10]); \
            MMA_INST(_C[_C_off - 13], _C[_C_off - 12], _b0, _A[9]);  \
            MMA_INST(_C[_C_off - 15], _C[_C_off - 14], _b0, _A[8]);  \
            MMA_INST(_C[_C_off - 17], _C[_C_off - 16], _b0, _A[7]);  \
            MMA_INST(_C[_C_off - 19], _C[_C_off - 18], _b0, _A[6]);  \
            MMA_INST(_C[_C_off - 21], _C[_C_off - 20], _b0, _A[5]);  \
            MMA_INST(_C[_C_off - 23], _C[_C_off - 22], _b0, _A[4]);  \
            MMA_INST(_C[_C_off - 25], _C[_C_off - 24], _b0, _A[3]);  \
            MMA_INST(_C[_C_off - 27], _C[_C_off - 26], _b0, _A[2]);  \
            MMA_INST(_C[_C_off - 29], _C[_C_off - 28], _b0, _A[1]);  \
            MMA_INST(_C[_C_off - 31], _C[_C_off - 30], _b0, _A[0]);  \
        }

#define MMA_INST_1x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  _B[0], _A); \
        }

#define MMA_INST_2x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  _B[0], _A); \
        }

#define MMA_INST_4x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  _B[0], _A); \
        }

#define MMA_INST_8x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND8  (_C, 0,  _B[0], _A); \
        }

#define MMA_INST_16x1(_C, _B, _A) \
        { \
            MMA_INST_ASCEND16 (_C, 0,  _B[0], _A); \
        }

#define MMA_INST_1x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND1 (_C, 3,  _B[1], _A); \
        }

#define MMA_INST_2x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND2 (_C, 7,  _B[1], _A); \
        }

#define MMA_INST_4x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND4 (_C, 15, _B[1], _A); \
        }

#define MMA_INST_8x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND8  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND8 (_C, 31, _B[1], _A); \
        }

#define MMA_INST_16x2(_C, _B, _A) \
        { \
            MMA_INST_ASCEND16 (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND16(_C, 63, _B[1], _A); \
        }

#define MMA_INST_1x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND1 (_C, 3,  _B[1], _A); \
            \
            MMA_INST_ASCEND1  (_C, 4,  _B[2], _A); \
            MMA_INST_DESCEND1 (_C, 7,  _B[3], _A); \
        }

#define MMA_INST_2x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND2 (_C, 7,  _B[1], _A); \
            \
            MMA_INST_ASCEND2  (_C, 8,  _B[2], _A); \
            MMA_INST_DESCEND2 (_C, 15, _B[3], _A); \
        }

#define MMA_INST_4x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND4 (_C, 15, _B[1], _A); \
            \
            MMA_INST_ASCEND4  (_C, 16, _B[2], _A); \
            MMA_INST_DESCEND4 (_C, 31, _B[3], _A); \
        }

#define MMA_INST_8x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND8  (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND8 (_C, 31, _B[1], _A); \
            \
            MMA_INST_ASCEND8  (_C, 32, _B[2], _A); \
            MMA_INST_DESCEND8 (_C, 63, _B[3], _A); \
        }

#define MMA_INST_16x4(_C, _B, _A) \
        { \
            MMA_INST_ASCEND16 (_C, 0,  _B[0], _A); \
            MMA_INST_DESCEND16(_C, 63, _B[1], _A); \
            \
            MMA_INST_ASCEND16 (_C, 64, _B[2], _A); \
            MMA_INST_DESCEND16(_C, 127,_B[3], _A); \
        }

#define MMA_INST_1x8(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  _B[0],  _A); \
            MMA_INST_DESCEND1 (_C, 3,  _B[1],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 4,  _B[2],  _A); \
            MMA_INST_DESCEND1 (_C, 7,  _B[3],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 8,  _B[4],  _A); \
            MMA_INST_DESCEND1 (_C, 11, _B[5],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 12, _B[6],  _A); \
            MMA_INST_DESCEND1 (_C, 15, _B[7],  _A); \
        }

#define MMA_INST_2x8(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  _B[0],  _A); \
            MMA_INST_DESCEND2 (_C, 7,  _B[1],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 8,  _B[2],  _A); \
            MMA_INST_DESCEND2 (_C, 15, _B[3],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 16, _B[4],  _A); \
            MMA_INST_DESCEND2 (_C, 23, _B[5],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 24, _B[6],  _A); \
            MMA_INST_DESCEND2 (_C, 31, _B[7],  _A); \
        }

#define MMA_INST_4x8(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,  _B[0],  _A); \
            MMA_INST_DESCEND4 (_C, 15, _B[1],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 16, _B[2],  _A); \
            MMA_INST_DESCEND4 (_C, 31, _B[3],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 32, _B[4],  _A); \
            MMA_INST_DESCEND4 (_C, 47, _B[5],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 48, _B[6],  _A); \
            MMA_INST_DESCEND4 (_C, 63, _B[7],  _A); \
        }

#define MMA_INST_8x8(_C, _B, _A) \
        { \
            MMA_INST_ASCEND8  (_C, 0,   _B[0], _A); \
            MMA_INST_DESCEND8 (_C, 31,  _B[1], _A); \
            \
            MMA_INST_ASCEND8  (_C, 32,  _B[2], _A); \
            MMA_INST_DESCEND8 (_C, 63,  _B[3], _A); \
	    \
            MMA_INST_ASCEND8  (_C, 64,  _B[4], _A); \
            MMA_INST_DESCEND8 (_C, 95,  _B[5], _A); \
            \
            MMA_INST_ASCEND8  (_C, 96,  _B[6], _A); \
            MMA_INST_DESCEND8 (_C, 127, _B[7], _A); \
        }

#define MMA_INST_1x16(_C, _B, _A) \
        { \
            MMA_INST_ASCEND1  (_C, 0,  _B[0],  _A); \
            MMA_INST_DESCEND1 (_C, 3,  _B[1],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 4,  _B[2],  _A); \
            MMA_INST_DESCEND1 (_C, 7,  _B[3],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 8,  _B[4],  _A); \
            MMA_INST_DESCEND1 (_C, 11, _B[5],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 12, _B[6],  _A); \
            MMA_INST_DESCEND1 (_C, 15, _B[7],  _A); \
	        \
            MMA_INST_ASCEND1  (_C, 16, _B[8],  _A); \
            MMA_INST_DESCEND1 (_C, 19, _B[9],  _A); \
            \
            MMA_INST_ASCEND1  (_C, 20, _B[10], _A); \
            MMA_INST_DESCEND1 (_C, 23, _B[11], _A); \
            \
            MMA_INST_ASCEND1  (_C, 24, _B[12], _A); \
            MMA_INST_DESCEND1 (_C, 27, _B[13], _A); \
            \
            MMA_INST_ASCEND1  (_C, 28, _B[14], _A); \
            MMA_INST_DESCEND1 (_C, 31, _B[15], _A); \
        }

#define MMA_INST_2x16(_C, _B, _A) \
        { \
            MMA_INST_ASCEND2  (_C, 0,  _B[0],  _A); \
            MMA_INST_DESCEND2 (_C, 7,  _B[1],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 8,  _B[2],  _A); \
            MMA_INST_DESCEND2 (_C, 15, _B[3],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 16, _B[4],  _A); \
            MMA_INST_DESCEND2 (_C, 23, _B[5],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 24, _B[6],  _A); \
            MMA_INST_DESCEND2 (_C, 31, _B[7],  _A); \
	        \
            MMA_INST_ASCEND2  (_C, 32, _B[8],  _A); \
            MMA_INST_DESCEND2 (_C, 39, _B[9],  _A); \
            \
            MMA_INST_ASCEND2  (_C, 40, _B[10], _A); \
            MMA_INST_DESCEND2 (_C, 47, _B[11], _A); \
            \
            MMA_INST_ASCEND2  (_C, 48, _B[12], _A); \
            MMA_INST_DESCEND2 (_C, 55, _B[13], _A); \
            \
            MMA_INST_ASCEND2  (_C, 56, _B[14], _A); \
            MMA_INST_DESCEND2 (_C, 63, _B[15], _A); \
        }

#define MMA_INST_4x16(_C, _B, _A) \
        { \
            MMA_INST_ASCEND4  (_C, 0,   _B[0],  _A); \
            MMA_INST_DESCEND4 (_C, 15,  _B[1],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 16,  _B[2],  _A); \
            MMA_INST_DESCEND4 (_C, 31,  _B[3],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 32,  _B[4],  _A); \
            MMA_INST_DESCEND4 (_C, 47,  _B[5],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 48,  _B[6],  _A); \
            MMA_INST_DESCEND4 (_C, 63,  _B[7],  _A); \
	        \
            MMA_INST_ASCEND4  (_C, 64,  _B[8],  _A); \
            MMA_INST_DESCEND4 (_C, 79,  _B[9],  _A); \
            \
            MMA_INST_ASCEND4  (_C, 80,  _B[10], _A); \
            MMA_INST_DESCEND4 (_C, 95,  _B[11], _A); \
            \
            MMA_INST_ASCEND4  (_C, 96,  _B[12], _A); \
            MMA_INST_DESCEND4 (_C, 111, _B[13], _A); \
            \
            MMA_INST_ASCEND4  (_C, 112, _B[14], _A); \
            MMA_INST_DESCEND4 (_C, 127, _B[15], _A); \
        }

