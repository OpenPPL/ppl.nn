#!/usr/bin/env python3

"""generate swzl conv kernels dynamically
"""

import os
import sys
import hashlib

class KernelInfo:
    def __init__(self, path, flt_size, k_size, cta_y_num, cta_x_num, warp_y, warp_x, buf_size):
        self.path = path
        self.flt_size = flt_size

        self.k_size = k_size

        self.cta_y_num = cta_y_num
        self.cta_x_num = cta_x_num

        self.warp_y = warp_y
        self.warp_x = warp_x

        self.buf_size = buf_size

        self.cta_y = self.cta_y_num * self.warp_y
        self.cta_x = self.cta_x_num * self.warp_x

        self.kconfig = "_b" + str(self.cta_y) + "x" + str(self.cta_x) + \
                "_w" + str(self.warp_y) + "x" + str(self.warp_x) + \
                "_k" + str(self.k_size) + "_buf" + str(self.buf_size)

        self.kname = "nvSwzlConv_hmma1688_nhwc_" + self.flt_size + self.kconfig
        self.fname = self.flt_size + "/swzl_"  + self.flt_size + self.kconfig + ".cu"

        self.WARP_SIZE = 32
        self.INT4_TO_8HALF = 8
        self.MMA_X = 16
        self.MMA_Y = 8
        self.MMA_X_HALF = self.MMA_X / 2

        self.cta_num = cta_y_num * cta_x_num
        self.cta_size = self.cta_num * self.WARP_SIZE

        self.sAv4_size = warp_y / self.INT4_TO_8HALF
        self.sBv4_size = warp_x / self.INT4_TO_8HALF

        self.dAv4_size = (self.cta_y * self.k_size * 1.0) / (self.INT4_TO_8HALF * self.cta_size)
        self.dBv4_size = (self.cta_x * self.k_size * 1.0) / (self.INT4_TO_8HALF * self.cta_size)

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
        elif size == 32.0:
            return str("32")
        else:
            sys.exit(1)

    def ExceedScope(self):
        MAX_SMEM_V4_PER_CTA = 3072 # 48KB per cta

        sm_a_v4 = self.cta_y * self.k_size * self.buf_size / self.INT4_TO_8HALF
        sm_b_v4 = self.cta_x * self.k_size * self.buf_size / self.INT4_TO_8HALF
        sm_r_v4 = self.cta_y * self.MMA_X  / self.INT4_TO_8HALF

        return max(sm_a_v4 + sm_b_v4, sm_r_v4) > MAX_SMEM_V4_PER_CTA

    def GenKernel(self):
        f = open(os.path.join(self.path, self.fname), "w")

        f.write("#define TILE_N_PER_CTA       %d\n" % self.cta_x)
        f.write("#define TILE_M_PER_CTA       %d\n\n" % self.cta_y)

        f.write("#define TILE_N_PER_WARP      %d\n" % self.warp_x)
        f.write("#define TILE_M_PER_WARP      %d\n\n" % self.warp_y)

        f.write("#define TILE_K_PER_CTA       %d\n" % self.k_size)
        f.write("#define TILE_K_PER_WARP      %d\n\n" % self.k_size)

        if self.warp_y == 8:
            f.write("#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)                 READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read)\n")
            f.write("#define WRITE_sRv1(_sm_base_v1, _sRv1_write_base, _C, _C_off)    WRITE_sRv1_%dx4(_sm_base_v1, _sRv1_write_base, _C, _C_off)\n\n" % (self.warp_y / self.MMA_Y))
        elif self.warp_y == 16:
            f.write("#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)                 READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read)\n")
            f.write("#define WRITE_sRv1(_sm_base_v1, _sRv1_write_base, _C, _C_off)    WRITE_sRv1_%dx2(_sm_base_v1, _sRv1_write_base, _C, _C_off)\n\n" % (self.warp_y / self.MMA_Y))
        elif self.warp_y == 32:
            f.write("#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)                 READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read)\n")
            f.write("#define WRITE_sRv1(_sm_base_v1, _sRv1_write_base, _C, _C_off)    WRITE_sRv1_%dx2(_sm_base_v1, _sRv1_write_base, _C, _C_off)\n\n" % (self.warp_y / self.MMA_Y))
        elif self.warp_y == 64:
            f.write("#define READ_sRv4(_Rv4, _sm_base_v4, _sRv4_read)                 READ_sRv4_SIZE4(_Rv4, _sm_base_v4, _sRv4_read)\n")
            f.write("#define WRITE_sRv1(_sm_base_v1, _sRv1_write_base, _C, _C_off)    WRITE_sRv1_%dx2(_sm_base_v1, _sRv1_write_base, _C, _C_off)\n\n" % (self.warp_y / self.MMA_Y))
        else:
            """warp_y error"""
            sys.exit(1)


        f.write("#define KERNEL_NAME %s\n\n" % self.kname)

        f.write("#define USE_%dBUF\n\n" % self.buf_size)

        f.write("#include <cuda_fp16.h>\n\n")

        f.write("#include \"swzl/common/const_macros.h\"\n\n")
        f.write("#include \"swzl/%s/bound_macros.h\"\n\n" % self.flt_size)
        f.write("#include \"swzl/common/ldsm_macros.h\"\n\n")
        f.write("#include \"swzl/%s/dmem_macros.h\"\n\n" % self.flt_size)
        f.write("#include \"swzl/common/hmma_macros.h\"\n\n")
        f.write("#include \"swzl/common/reduce_macros.h\"\n\n")
        f.write("#include \"swzl/common/smem_macros.h\"\n\n")

        f.write("#define MMA_INSTS(_C, _B, _A)           MMA_INST_%dx%d(_C, _B, _A)\n\n" % (self.warp_y / self.MMA_Y, self.warp_x / self.MMA_X))

        f.write("#define READ_sAv1(_A, _sm_base_v1, _sAv1_read)          READ_sUv1_1x%d(_A, _sm_base_v1, _sAv1_read)\n" % (self.warp_y / self.MMA_Y))

        if self.warp_x == 128:
            f.write("#define READ_sBv1(_B, _sm_base_v1, _sBv1_read)          READ_sUv1_%dx%d(_B, _sm_base_v1, _sBv1_read)\n\n" % (4, self.warp_x / self.MMA_X_HALF / 4))
        elif self.warp_x == 64:
            f.write("#define READ_sBv1(_B, _sm_base_v1, _sBv1_read)          READ_sUv1_%dx%d(_B, _sm_base_v1, _sBv1_read)\n\n" % (2, self.warp_x / self.MMA_X_HALF / 2))
        elif self.warp_x == 16 or self.warp_x == 32:
            f.write("#define READ_sBv1(_B, _sm_base_v1, _sBv1_read)          READ_sUv1_%dx%d(_B, _sm_base_v1, _sBv1_read)\n\n" % (1, self.warp_x / self.MMA_X_HALF))
        else:
            """warp_x error"""
            sys.exit(1)

        if self.flt_size == "f1":
            f.write("#define LOAD_dAv4(_regA, _dA, _dAv4_off, _flt_c_v8_id, _flt_n_valid)    LOAD_dAv4_SIZE%s(_regA, _dA, _dAv4_off, _flt_c_v8_id, _flt_n_valid)\n" % self.GetSizeString(self.dAv4_size))
            f.write("#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dAv4_size))

            f.write("#define LOAD_dBv4(_regB, _dB, _dBv4_off, _in_c_v8_id, _in_hw_valid)     LOAD_dBv4_SIZE%s(_regB, _dB, _dBv4_off, _in_c_v8_id, _in_hw_valid)\n" % self.GetSizeString(self.dBv4_size))
            f.write("#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dBv4_size))

            f.write("#define FLT_SIZE1\n\n")

        elif self.flt_size == "f3":
            f.write("#define LOAD_dAv4(_regA, _dA, _dAv4_off, _flt_c_v8_id, _flt_n_valid)    LOAD_dAv4_SIZE%s(_regA, _dA, _dAv4_off, _flt_c_v8_id, _flt_n_valid)\n" % self.GetSizeString(self.dAv4_size))
            f.write("#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dAv4_size))

            f.write("#define LOAD_dBv4(_regB, _dB, _dBv4_off, _in_c_v8_id, _flt_hw_bid)      LOAD_dBv4_SIZE%s(_regB, _dB, _dBv4_off, _in_c_v8_id, _flt_hw_bid)\n" % self.GetSizeString(self.dBv4_size))
            f.write("#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dBv4_size))

            f.write("#define FLT_SIZE3\n\n")
        elif self.flt_size == "fn":
            f.write("#define LOAD_dAv4(_regA, _dA, _dAv4_off, _flt_c_v8_id, _flt_n_valid)    LOAD_dAv4_SIZE%s(_regA, _dA, _dAv4_off, _flt_c_v8_id, _flt_n_valid)\n" % self.GetSizeString(self.dAv4_size))
            f.write("#define WRITE_sAv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dAv4_size))

            f.write("#define LOAD_dBv4(_regB, _dB, _dBv4_off, _in_n_id, _in_h_id, _in_w_id)  LOAD_dBv4_SIZE%s(_regB, _dB, _dBv4_off, _in_n_id, _in_h_id, _in_w_id)\n" % self.GetSizeString(self.dBv4_size))
            f.write("#define WRITE_sBv4(_sm_base_v4, _sm_off, _reg)                          WRITE_sUv4_SIZE%s(_sm_base_v4, _sm_off, _reg)\n\n" % self.GetSizeString(self.dBv4_size))

            f.write("#define FWD_FLT(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid)    FWD_FLT_SIZE%s(_flt_h_id, _flt_w_id, _flt_c_v8_id, _flt_c_v8_valid)\n" % (self.GetSizeString(self.dAv4_size) if self.dAv4_size >= 1 else "1"))

            f.write("#define FLT_SIZEN\n\n")

        f.write("#include \"swzl/common/output_macros.h\"\n\n")

        f.write("#include \"swzl/common/main_body.h\"\n\n")

        f.write("#include \"swzl/common/uni_undefs.h\"\n\n")

        f.close()

class LutSourceFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_lut_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include  \"swzl/%s_lut_kernels.h\"\n\n" % flt_size)

        self.f.write("#define ENABLE_FUSE\n\n")

    def AppendKernel(self, fname):
        self.f.write("#include \"swzl/%s\"\n" % fname)

    def Close(self):
        self.f.close()

class SpkSourceFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_spk_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include  \"swzl/%s_spk_kernels.h\"\n\n" % flt_size)

        self.f.write("#define ENABLE_SPLITK\n\n")

    def AppendKernel(self, fname):
        self.f.write("#include \"swzl/%s\"\n" % fname)

    def Close(self):
        self.f.close()

class LutHeaderFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_lut_kernels.h"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#ifndef __PPLCUDA_SWZL_%s_LUT_KERNELS_H__\n" % flt_size.upper())
        self.f.write("#define __PPLCUDA_SWZL_%s_LUT_KERNELS_H__\n" % flt_size.upper())

        self.f.write("\n\n#include \"kernel_type.h\"\n\n")

    def AppendKernel(self, kname):
        self.f.write("__global__ lut_kernel_t %s;\n" % kname)

    def Close(self):
        self.f.write("\n\n#endif\n")
        self.f.close()

class SpkHeaderFile:
    def __init__(self, path, flt_size):
        self.path = path
        self.flt_size = flt_size

        self.fname = flt_size + "_spk_kernels.h"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#ifndef __PPLCUDA_SWZL_%s_SPK_KERNELS_H__\n" % flt_size.upper())
        self.f.write("#define __PPLCUDA_SWZL_%s_SPK_KERNELS_H__\n" % flt_size.upper())

        self.f.write("\n\n#include \"kernel_type.h\"\n\n")

    def AppendKernel(self, kname):
        self.f.write("__global__ spk_kernel_t %s;\n" % kname)

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

        self.f.write("#include \"swzl/%s_lut_kernels.h\"\n" % self.flt_size)
        self.f.write("#include \"swzl/%s_spk_kernels.h\"\n\n" % self.flt_size)

        self.f.write("void InitializeSwzlConv%sKernelContainer(std::vector<kernel_info_t> & kernel_container)\n{\n" % self.flt_size.upper())

    def AppendKernel(self, kname):
        self.f.write("\tADD_KERNEL(CONV_SWZL_%s, \"%s\", &%s, &%s, NULL);\n" % (self.flt_size.upper(), kname, kname, kname))

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

    for flt_size in ["f1", "f3", "fn"]:
        init_file = InitFile(parent_path, flt_size)

        path = parent_path + '/' + flt_size

        if not os.path.exists(path):
            os.makedirs(path)

        lut_header_file = LutHeaderFile(parent_path, flt_size)
        spk_header_file = SpkHeaderFile(parent_path, flt_size)

        lut_source_file = LutSourceFile(parent_path, flt_size)
        spk_source_file = SpkSourceFile(parent_path, flt_size)

        for buf_size in [1, 2]:
            for k_size in [8, 16, 32, 64]:
                for warp_y in [8, 16, 32, 64]:
                    for warp_x in [16, 32, 64, 128]:
                        for cta_y_num in [1, 2, 4]:
                            for cta_x_num in [1, 2, 4]:
                                if warp_y == 8 and warp_x == 16:
                                    continue
                                if warp_y == 64 and warp_x == 128:
                                    continue
                                if cta_y_num == 1 and cta_x_num == 1 and k_size == 64:
                                    continue
                                if cta_y_num == 4 and cta_x_num == 4:
                                    continue

                                kernel = KernelInfo(parent_path, flt_size, k_size, cta_y_num, cta_x_num, warp_y, warp_x, buf_size)

                                if not kernel.ExceedScope():
                                    kernel.GenKernel()
                                    lut_header_file.AppendKernel(kernel.kname)
                                    spk_header_file.AppendKernel(kernel.kname)

                                    lut_source_file.AppendKernel(kernel.fname)
                                    spk_source_file.AppendKernel(kernel.fname)

                                    init_file.AppendKernel(kernel.kname)
        lut_header_file.Close()
        spk_header_file.Close()

        lut_source_file.Close()
        spk_source_file.Close()

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
