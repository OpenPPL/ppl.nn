#!/usr/bin/env python3

"""generate 2spk conv kernels dynamically
"""

import os
import sys
import hashlib

def CeilDiv(x, y):
    return -(x // -y)

class KernelInfo:
    def __init__(self, path, flt_size, s_size, k_num, cta_y_num, cta_x_num, warp_y, warp_x, buf_size):
        self.path = path
        self.flt_size = flt_size

        self.s_size = s_size
        self.k_num = k_num
        self.k_size = self.k_num * self.s_size

        self.cta_y_num = cta_y_num
        self.cta_x_num = cta_x_num

        self.warp_y = warp_y
        self.warp_x = warp_x

        self.buf_size = buf_size

        self.cta_y = self.cta_y_num * self.warp_y
        self.cta_x = self.cta_x_num * self.warp_x

        self.kconfig = "_b" + str(self.cta_y) + "x" + str(self.cta_x) + \
                "_w" + str(self.warp_y) + "x" + str(self.warp_x) + \
                "_k" + str(self.k_size) + "_s" + str(self.s_size) + "_buf" + str(self.buf_size)

        self.kname = "nv2spkSm75Int8Conv_imma8816_nhwc_" + self.flt_size + self.kconfig
        self.fname = self.flt_size + "/2spk_"  + self.flt_size + self.kconfig + ".cu"

        self.CHAR_SIZE = 4
        self.WARP_SIZE = 32
        self.WARP_SIZE_Y = 8
        self.WARP_SIZE_X = 4
        self.INT_TO_4CHAR = 4
        self.INT4_TO_4INT = 4
        self.INT4_TO_16BYTE = 16
        self.MMA_Y = 8
        self.MMA_K = 16
        self.MMA_X = 8
        self.MMA_Y_HALF = self.MMA_Y / 2
        self.PB_PER_TURING_SM = 4

        self.CPI_IMMA8816 = 8.06
        self.CPI_L1_LDG128 = 8
        self.IMMA_LATENCY = 14
        self.LDSM1_LATENCY = 19
        self.STS128_LATENCY = 23
        self.DRAM_LATENCY = 220

        self.MAX_REG_NUM_PER_THD = 255
        self.MAX_REG_NUM_PER_CTA = 65536

        # TODO: fix here for T4 and A100
        #self.MAX_SMEM_V4_PER_CTA = 48  * (1024 >> 4) # 48KB per cta
        self.MAX_SMEM_V4_PER_CTA = 163 * (1024 >> 4) # 163KB per cta

        self.thd_y = self.warp_y // self.WARP_SIZE_Y
        self.thd_x = self.warp_x // self.WARP_SIZE_X

        self.cta_num = cta_y_num * cta_x_num
        self.cta_size = self.cta_num * self.k_num * self.WARP_SIZE

        self.thd_num_per_set = self.cta_num * self.WARP_SIZE

        self.sAv4_size = warp_y / (self.INT4_TO_4INT * self.INT_TO_4CHAR)
        self.sBv4_size = warp_x / (self.INT4_TO_4INT * self.INT_TO_4CHAR)

        self.dAv4_size = (self.cta_y * self.k_size * 1.0) / (self.INT4_TO_4INT * self.INT_TO_4CHAR * self.cta_size)
        self.dBv4_size = (self.cta_x * self.k_size * 1.0) / (self.INT4_TO_4INT * self.INT_TO_4CHAR * self.cta_size)

    def GetSizeString(self, size):
        if size == 0.0625:
            return str("_16TH")
        elif size == 0.125:
            return str("_8TH")
        elif size == 0.25:
            return str("_QTR")
        elif size == 0.5:
            return str("_HALF")
        elif size == 1.0:
            return str("1")
        elif size == 2.0:
            return str("2")
        elif size == 4.0:
            return str("4")
        elif size == 8.0:
            return str("8")
        elif size == 16.0:
            return str("16")
        else:
            sys.exit(1)

    def GetSMemUsage(self):

        sm_a_v4 = self.cta_y * self.k_size * self.buf_size / (self.INT4_TO_4INT * self.INT_TO_4CHAR)
        sm_b_v4 = self.cta_x * self.k_size * self.buf_size / (self.INT4_TO_4INT * self.INT_TO_4CHAR)
        sm_c_v4 = self.cta_y * self.cta_x  / self.INT4_TO_4INT

        return max(sm_a_v4 + sm_b_v4, sm_c_v4 * self.k_num)

    def GetRegUsage(self):
        ret = 0

        reg_a_v4 = CeilDiv(self.cta_y * self.k_size, (self.CHAR_SIZE * self.INT4_TO_4INT * self.cta_size))
        reg_b_v4 = CeilDiv(self.cta_x * self.k_size, (self.CHAR_SIZE * self.INT4_TO_4INT * self.cta_size))
        reg_c_v4 = CeilDiv(self.cta_y * self.cta_x,  (self.INT4_TO_4INT * self.thd_num_per_set))

        reg_a_v1 = reg_a_v4 * self.INT4_TO_4INT
        reg_b_v1 = reg_b_v4 * self.INT4_TO_4INT
        reg_c_v1 = reg_c_v4 * self.INT4_TO_4INT

        reg_a_buf_v1 = self.thd_y * self.buf_size
        reg_b_buf_v1 = self.thd_x // self.CHAR_SIZE * self.buf_size

        reg_a_idx = reg_a_v4 * 2
        reg_b_idx = reg_b_v4 * 2

        reg_reduce_v1 = self.k_num * self.INT4_TO_4INT

        reg_common_idx = 40

        ret = reg_a_v1 + reg_b_v1 + reg_c_v1 + reg_a_buf_v1 + reg_b_buf_v1 + reg_a_idx + reg_b_idx + reg_reduce_v1 + reg_common_idx

        return ret

    def GetCompMem2Ratio(self):
        pb_num_per_cta = self.cta_num if self.cta_num < self.PB_PER_TURING_SM else self.PB_PER_TURING_SM

        cycles_comp = self.CPI_IMMA8816 * ( (self.cta_y / self.MMA_Y) * (self.cta_x / self.MMA_X) * (self.k_size / self.MMA_K) / pb_num_per_cta) + self.IMMA_LATENCY + self.LDSM1_LATENCY

        cycles_mem2 = self.CPI_L1_LDG128 * CeilDiv( (self.cta_y + self.cta_x) * self.k_size * self.CHAR_SIZE, (self.INT4_TO_16BYTE * self.WARP_SIZE) ) + self.DRAM_LATENCY + self.STS128_LATENCY

        comp_mem2_ratio = cycles_comp / cycles_mem2

        return comp_mem2_ratio

    def IsKernelFeasible(self):
        if self.cta_size > 512 or self.cta_size < 64:
            return False

        reg_usage_per_thd = self.GetRegUsage()
        reg_usage_per_cta = reg_usage_per_thd * self.cta_size
        if reg_usage_per_thd > self.MAX_REG_NUM_PER_THD or reg_usage_per_cta > self.MAX_REG_NUM_PER_CTA:
            return False

        smem_usage = self.GetSMemUsage()
        if smem_usage > self.MAX_SMEM_V4_PER_CTA:
            return False

        comp_mem2_ratio = self.GetCompMem2Ratio()
        if comp_mem2_ratio >= 2 or comp_mem2_ratio <= 0.5:
            return False

        return True


    def GenKernel(self):
        f = open(os.path.join(self.path, self.fname), "w")

        f.write("#define TILE_N_PER_MMA       %d\n" % self.MMA_X)
        f.write("#define TILE_K_PER_MMA       %d\n" % self.MMA_K)
        f.write("#define TILE_M_PER_MMA       %d\n\n" % self.MMA_Y)

        f.write("#define TILE_N_PER_CTA       %d\n" % self.cta_x)
        f.write("#define TILE_M_PER_CTA       %d\n\n" % self.cta_y)

        f.write("#define TILE_N_PER_WARP      %d\n" % self.warp_x)
        f.write("#define TILE_M_PER_WARP      %d\n\n" % self.warp_y)

        f.write("#define TILE_K_PER_CTA       %d\n" % self.k_size)
        f.write("#define TILE_K_PER_SET       %d\n" % self.s_size)
        f.write("#define TILE_K_PER_WARP      %d\n\n" % self.s_size)

        f.write("#define INTER_SET_REDUCE_RATIO  ((TILE_K_PER_CTA) / (TILE_K_PER_SET))\n\n")

        if self.k_num == 2:
            f.write("#define REDUCE(_R)              REDUCE_INT_1x4(_R)\n\n")
            f.write("#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)        READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read)\n\n")
        elif self.k_num == 4:
            f.write("#define REDUCE(_R)              REDUCE_INT_3x4(_R)\n\n")
            f.write("#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)        READ_sRv4_SIZE4(_Rv4, _sm_base_v4, _sRv4_read)\n\n")
        elif self.k_num == 1:
            f.write("#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)        READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read)\n\n")
        else:
            """knum error"""
            sys.exit(1)

        f.write("#define KERNEL_NAME %s\n\n" % self.kname)

        f.write("#define BUF_NUM %d\n\n" % self.buf_size)

        f.write("#define USE_IMMA8816\n\n")

        f.write("#include \"2spk/int8/const_macros.h\"\n\n")
        f.write("#include \"2spk/int8/%s/bound_macros.h\"\n\n" % self.flt_size)
        f.write("#include \"2spk/int8/ldsm_macros.h\"\n\n")

        if self.buf_size <= 2:
            f.write("#include \"2spk/int8/%s/dmem_reg_macros.h\"\n\n" % self.flt_size)
        elif self.buf_size > 2:
            f.write("#include \"2spk/int8/async_macros.h\"\n\n")
            f.write("#include \"2spk/int8/%s/dmem_async_macros.h\"\n\n" % self.flt_size)

        f.write("#include \"2spk/int8/imma8816_macros.h\"\n\n")
        f.write("#include \"2spk/int8/reduce_macros.h\"\n\n")
        f.write("#include \"2spk/int8/smem_macros.h\"\n\n")

        f.write("#define MMA_INSTS(_C, _A, _B)           MMA_INST_%dx%d(_C, _A, _B)\n\n" % (self.warp_y / self.MMA_Y, self.warp_x / self.MMA_X))

        f.write("#define READ_sAv1(_A, _sm_base_v1, _sAv1_read)          READ_sUv1_K1_1x%d(_A, _sm_base_v1, _sAv1_read)\n"   % (self.warp_y / self.MMA_Y))
        f.write("#define READ_sBv1(_B, _sm_base_v1, _sBv1_read)          READ_sUv1_K1_1x%d(_B, _sm_base_v1, _sBv1_read)\n\n" % (self.warp_x / self.MMA_X))

        f.write("#define WRITE_sRv2(_sm_base_v2, _sRv2_write_base, _C)   WRITE_sRv2_%dx%d(_sm_base_v2, _sRv2_write_base, _C)\n\n" % (self.warp_y / self.MMA_Y, self.warp_x /self.MMA_X))

        if self.buf_size <= 2:
            if self.flt_size == "f1" or self.flt_size == "fs":
                f.write("#define LOAD_dAv4(_regA, _dA, _dAv4_off, _in_c_v16_id, _in_hw_valid)    LOAD_dAv4_SIZE%s(_regA, _dA, _dAv4_off, _in_c_v16_id, _in_hw_valid)\n" % self.GetSizeString(self.dAv4_size))
                f.write("#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dAv4_size))

                f.write("#define LOAD_dBv4(_regB, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)   LOAD_dBv4_SIZE%s(_regB, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)\n" % self.GetSizeString(self.dBv4_size))
                f.write("#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dBv4_size))

                f.write("#define FLT_SIZE1\n\n")

            elif self.flt_size == "f3":
                f.write("#define LOAD_dAv4(_regA, _dA, _dAv4_off, _in_c_v16_id, _flt_hw_bid)     LOAD_dAv4_SIZE%s(_regA, _dA, _dAv4_off, _in_c_v16_id, _flt_hw_bid)\n" % self.GetSizeString(self.dAv4_size))
                f.write("#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dAv4_size))

                f.write("#define LOAD_dBv4(_regB, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)   LOAD_dBv4_SIZE%s(_regB, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)\n" % self.GetSizeString(self.dBv4_size))
                f.write("#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dBv4_size))

                f.write("#define FLT_SIZE3\n\n")
            elif self.flt_size == "fn":
                f.write("#define LOAD_dAv4(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id)  LOAD_dAv4_SIZE%s(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id)\n" % self.GetSizeString(self.dAv4_size))
                f.write("#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dAv4_size))

                f.write("#define LOAD_dBv4(_regB, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)   LOAD_dBv4_SIZE%s(_regB, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)\n" % self.GetSizeString(self.dBv4_size))
                f.write("#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dBv4_size))

                f.write("#define FWD_FLT(_flt_h_id, _flt_w_id, _flt_c_v16_id, _flt_c_v16_valid)    FWD_FLT_SIZE%s(_flt_h_id, _flt_w_id, _flt_c_v16_id, _flt_c_v16_valid)\n" % (self.GetSizeString(self.dAv4_size) if self.dAv4_size >= 1 else "1"))

                f.write("#define FLT_SIZEN\n\n")
        elif self.buf_size > 2:
            if self.flt_size == "f1" or self.flt_size == "fs":
                f.write("#define LOAD_dAv4(sAv4, _sAv4_off, _dA, _dAv4_off, _in_c_v16_id, _in_hw_valid)     LOAD_dAv4_SIZE%s(sAv4, _sAv4_off, _dA, _dAv4_off, _in_c_v16_id, _in_hw_valid)\n" % self.GetSizeString(self.dAv4_size))
                f.write("#define LOAD_dBv4(sBv4, _sBv4_off, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)    LOAD_dBv4_SIZE%s(sBv4, _sBv4_off, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)\n\n" % self.GetSizeString(self.dBv4_size))

                f.write("#define FLT_SIZE1\n\n")

            elif self.flt_size == "f3":
                f.write("#define LOAD_dAv4(sAv4, _sAv4_off, _dA, _dAv4_off, _in_c_v16_id, _flt_hw_bid)      LOAD_dAv4_SIZE%s(sAv4, _sAv4_off, _dA, _dAv4_off, _in_c_v16_id, _flt_hw_bid)\n" % self.GetSizeString(self.dAv4_size))
                f.write("#define LOAD_dBv4(sBv4, _sBv4_off, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)    LOAD_dBv4_SIZE%s(sBv4, _sBv4_off, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)\n\n" % self.GetSizeString(self.dBv4_size))

                f.write("#define FLT_SIZE3\n\n")
            elif self.flt_size == "fn":
                f.write("#define LOAD_dAv4(sAv4, _sAv4_off, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id)  LOAD_dAv4_SIZE%s(sAv4, _sAv4_off, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id)\n" % self.GetSizeString(self.dAv4_size))
                f.write("#define LOAD_dBv4(sBv4, _sBv4_off, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)   LOAD_dBv4_SIZE%s(sBv4, _sBv4_off, _dB, _dBv4_off, _flt_c_v16_id, _flt_n_valid)\n\n" % self.GetSizeString(self.dBv4_size))

                f.write("#define FWD_FLT(_flt_h_id, _flt_w_id, _flt_c_v16_id, _flt_c_v16_valid)    FWD_FLT_SIZE%s(_flt_h_id, _flt_w_id, _flt_c_v16_id, _flt_c_v16_valid)\n" % (self.GetSizeString(self.dAv4_size) if self.dAv4_size >= 1 else "1"))

                f.write("#define FLT_SIZEN\n\n")

            f.write("#define CP_ASYNC(_pred, _sm_v4, _sm_v4_off, _gm_v4, _gm_v4_off)                 PRED_CP_ASYNC_ZFILL_CG(_pred, _sm_v4 + _INT4_TO_16BYTE_ * (_sm_v4_off), _gm_v4 + _gm_v4_off)\n\n")

        f.write("#include \"2spk/int8/output_macros.h\"\n\n")

        f.write("#include \"2spk/int8/main_body8816.h\"\n\n")

        f.write("#include \"2spk/int8/uni_undefs.h\"\n\n")

        f.close()

class LutSourceFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_lut_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include  \"2spk/sm75/int8/imma8816/%s_lut_kernels.h\"\n\n" % flt_size)

        if self.flt_size == "f1" or self.flt_size == "f3" or self.flt_size == "fn":
            self.f.write("#define ENABLE_FUSE\n\n")
        elif self.flt_size == "fs":
            self.f.write("#define ENABLE_SPLITF\n\n")


    def AppendKernel(self, fname):
        self.f.write("#include \"2spk/sm75/int8/imma8816/%s\"\n" % fname)

    def Close(self):
        self.f.close()

class SpkSourceFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_spk_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include  \"2spk/sm75/int8/imma8816/%s_spk_kernels.h\"\n\n" % flt_size)

        if self.flt_size == "fs":
            self.f.write("#define ENABLE_SPLITF\n\n")

        self.f.write("#define ENABLE_SPLITK\n\n")

    def AppendKernel(self, fname):
        self.f.write("#include \"2spk/sm75/int8/imma8816/%s\"\n" % fname)

    def Close(self):
        self.f.close()

class LutHeaderFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_lut_kernels.h"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#ifndef __PPLCUDA_2SPK_SM75_INT8_IMMA8816_%s_LUT_KERNELS_H__\n" % flt_size.upper())
        self.f.write("#define __PPLCUDA_2SPK_SM75_INT8_IMMA8816_%s_LUT_KERNELS_H__\n" % flt_size.upper())

        self.f.write("\n\n#include \"kernel_type.h\"\n\n")

    def AppendKernel(self, kname):
        self.f.write("__global__ int8_lut_kernel_t %s;\n" % kname)

    def Close(self):
        self.f.write("\n\n#endif\n")
        self.f.close()

class SpkHeaderFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_spk_kernels.h"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#ifndef __PPLCUDA_2SPK_SM75_INT8_IMMA8816_%s_SPK_KERNELS_H__\n" % flt_size.upper())
        self.f.write("#define __PPLCUDA_2SPK_SM75_INT8_IMMA8816_%s_SPK_KERNELS_H__\n" % flt_size.upper())

        self.f.write("\n\n#include \"kernel_type.h\"\n\n")

    def AppendKernel(self, kname):
        self.f.write("__global__ int8_spk_kernel_t %s;\n" % kname)

    def Close(self):
        self.f.write("\n\n#endif\n")
        self.f.close()

class InitFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = "init_" + self.flt_size + "_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include \"conv_common.h\"\n\n")

        self.f.write("#include \"2spk/sm75/int8/imma8816/%s_lut_kernels.h\"\n" % self.flt_size)
        self.f.write("#include \"2spk/sm75/int8/imma8816/%s_spk_kernels.h\"\n\n" % self.flt_size)

        self.f.write("void Initialize2spkSM75Int8Imma8816Conv%sKernelContainer(std::vector<kernel_info_t> & kernel_container)\n{\n" % self.flt_size.upper())

    def AppendKernel(self, kname):
        self.f.write("\tADD_KERNEL(CONV_2SPK_%s, \"%s\", &%s, &%s, NULL);\n" % (self.flt_size.upper(), kname, kname, kname))

    def Close(self):
        self.f.write("\n}\n")
        self.f.close()

class HashFile:
    def __init__(self, path, hash_path):
        self.path = path
        self.fname = ".hash_file.txt"

        self.current_hash = dict()
        for root, dirs, files in os.walk(hash_path):
            for file in files:
                fname = os.path.join(root, file)
                fhash = hashlib.md5(open(fname, 'rb').read()).hexdigest()
                self.current_hash[fname] = fhash

    def CheckFileExist(self):
        return os.path.isfile(os.path.join(self.path, self.fname))

    def CompareWithPreviousHash(self):
        previous_hash = dict()

        for line in open(os.path.join(self.path, self.fname), "r"):
            fname, fhash = line.split()
            previous_hash[fname] = fhash

        return previous_hash == self.current_hash

    def WriteCurrentHash(self):
        self.f = open(os.path.join(self.path, self.fname), "w")

        for fname, fhash in self.current_hash.items():
            self.f.write("%s\t%s\n" % (fname, fhash))

    def Close(self):
        self.f.close()

def GenAllKernels(parent_path):

    for flt_size in ["f1", "f3", "fn", "fs"]:
        init_file = InitFile(parent_path, flt_size)

        path = parent_path + '/' + flt_size

        if not os.path.exists(path):
            os.makedirs(path)

        lut_header_file = LutHeaderFile(parent_path, flt_size)
        spk_header_file = SpkHeaderFile(parent_path, flt_size)

        lut_source_file = LutSourceFile(parent_path, flt_size)
        spk_source_file = SpkSourceFile(parent_path, flt_size)

        for buf_size in [1, 2]:
            for s_size in [16, 32, 64]:
                for k_num in [1, 2, 4]:
                    for warp_y in [16, 32, 64, 128]:
                        for warp_x in [8, 16, 32, 64]:
                            for cta_y_num in [1, 2, 4]:
                                for cta_x_num in [1, 2, 4]:
                                    if warp_y == 128 and warp_x == 64:
                                        continue
                                    if cta_y_num == 4 and cta_x_num == 4:
                                        continue
                                    if cta_y_num * cta_x_num * k_num == 1 and s_size * k_num == 128:
                                        continue
                                    if cta_y_num * cta_x_num * k_num <= 2 and s_size * k_num == 256:
                                        continue

                                    kernel = KernelInfo(parent_path, flt_size, s_size, k_num, cta_y_num, cta_x_num, warp_y, warp_x, buf_size)

                                    if kernel.IsKernelFeasible():
                                        kernel.GenKernel()
                                        lut_header_file.AppendKernel(kernel.kname)
                                        spk_header_file.AppendKernel(kernel.kname)

                                        lut_source_file.AppendKernel(kernel.fname)
                                        spk_source_file.AppendKernel(kernel.fname)

                                        init_file.AppendKernel(kernel.kname)
        lut_header_file.Close()
        spk_header_file.Close()

        lut_source_file.Close()
        lut_source_file.Close()

        init_file.Close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        os.makedirs(path)

    hash_file = HashFile(path, os.path.dirname(os.path.abspath(__file__)))

    if not hash_file.CheckFileExist() or not hash_file.CompareWithPreviousHash():

        GenAllKernels(path)

        hash_file.WriteCurrentHash()

        hash_file.Close()
