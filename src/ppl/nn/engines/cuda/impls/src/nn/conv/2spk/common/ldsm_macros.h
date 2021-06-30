////////////////////////////////////////
// ldsm macros
////////////////////////////////////////

#define LDSM_ROW_X1_OPCODE \
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"

#define LDSM_ROW_X2_OPCODE \
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"

#define LDSM_ROW_X4_OPCODE \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"

#define LDSM_ROW_X1_INST(_x0, _addr) \
        asm volatile(LDSM_ROW_X1_OPCODE:   "=r"(_x0)   : "r"(_addr));

#define LDSM_ROW_X2_INST(_x0, _x1, _addr) \
        asm volatile(LDSM_ROW_X2_OPCODE:   "=r"(_x0),   "=r"(_x1): "r"(_addr));

#define LDSM_ROW_X4_INST(_x0, _x1, _x2, _x3, _addr) \
        asm volatile(LDSM_ROW_X4_OPCODE:   "=r"(_x0),   "=r"(_x1),  "=r"(_x2),   "=r"(_x3): "r"(_addr));

#define LDSM_COL_X1_OPCODE \
        "ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"

#define LDSM_COL_X2_OPCODE \
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"

#define LDSM_COL_X4_OPCODE \
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"

#define LDSM_COL_X1_INST(_x0, _addr) \
        asm volatile(LDSM_COL_X1_OPCODE:   "=r"(_x0)   : "r"(_addr));

#define LDSM_COL_X2_INST(_x0, _x1, _addr) \
        asm volatile(LDSM_COL_X2_OPCODE:   "=r"(_x0),   "=r"(_x1): "r"(_addr));

#define LDSM_COL_X4_INST(_x0, _x1, _x2, _x3, _addr) \
        asm volatile(LDSM_COL_X4_OPCODE:   "=r"(_x0),   "=r"(_x1),  "=r"(_x2),   "=r"(_x3): "r"(_addr));
