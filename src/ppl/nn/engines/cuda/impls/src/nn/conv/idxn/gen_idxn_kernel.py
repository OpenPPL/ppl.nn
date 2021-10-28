#!/ usr / bin / env python3

"""generate idxn conv kernels dynamically
"""

import os
import sys
import hashlib

class KernelInfo:
    def __init__(self, path, s_size, k_num, cta_y_num, cta_x_num, warp_y, warp_x):
        self.path = path
        self.s_size = s_size
        self.k_num = k_num
        self.k_size = self.k_num * self.s_size

        self.cta_y_num = cta_y_num
        self.cta_x_num = cta_x_num

        self.warp_y = warp_y
        self.warp_x = warp_x

        self.cta_y = self.cta_y_num * self.warp_y
        self.cta_x = self.cta_x_num * self.warp_x

        self.kconfig = "_b" + str(self.cta_y) + "x" + str(self.cta_x) + \
                "_w" + str(self.warp_y) + "x" + str(self.warp_x) + \
                "_k" + str(self.k_size) + "_s" + str(self.s_size)

        self.kname = "nvIdxnConv_hmma1688_nhwc" + self.kconfig + "_nosmem"
        self.fname = "kernels" + "/idxn"  + self.kconfig + ".cu"

        self.WARP_SIZE = 32
        self.MMA_Y = 16
        self.MMA_X = 8
        self.MMA_Y_HALF = self.MMA_Y / 2

        self.cta_num = cta_y_num * cta_x_num
        self.cta_size = self.cta_num * self.WARP_SIZE

        self.dAvn_size = self.warp_y / self.MMA_Y_HALF
        self.dBvn_size = self.warp_x / self.MMA_X

    def GenKernel(self):
        f = open(os.path.join(self.path, self.fname), "w")

        f.write("#define TILE_N_PER_CTA       %d\n" % self.cta_x)
        f.write("#define TILE_M_PER_CTA       %d\n\n" % self.cta_y)

        f.write("#define TILE_N_PER_WARP      %d\n" % self.warp_x)
        f.write("#define TILE_M_PER_WARP      %d\n\n" % self.warp_y)

        f.write("#define TILE_K_PER_CTA       %d\n" % self.k_size)
        f.write("#define TILE_K_PER_STEP      %d\n\n" % self.s_size)

        f.write("#define KERNEL_NAME %s\n\n" % self.kname)

        f.write("#include <cuda_fp16.h>\n\n")

        f.write("#include \"idxn/common/const_macros.h\"\n\n")

        if self.s_size == 8:
            f.write("#include \"idxn/common/dmem_i1_macros.h\"\n\n")
            f.write("#include \"idxn/common/hmma_i1_macros.h\"\n\n")

            f.write("#define LOAD_dAv1(_regA, _dAv1, _in_id, _in_off)    LOAD_dAv1_SIZE%d(_regA, _dAv1, _in_id, _in_off)\n" % self.dAvn_size)
            f.write("#define LOAD_dBv1(_regB, _dBv1, _dBv1_off)          LOAD_dBv1_SIZE%d(_regB, _dBv1, _dBv1_off)\n\n" % self.dBvn_size)

            f.write("#define MMA_INSTS(_C, _A, _B)                       MMA_INST_1INT_%dx%d(_C, _A, _B)\n\n" % (self.dAvn_size / 2, self.dBvn_size))
        elif self.s_size == 16:
            f.write("#include \"idxn/common/dmem_i2_macros.h\"\n\n")
            f.write("#include \"idxn/common/hmma_i2_macros.h\"\n\n")

            f.write("#define LOAD_dAv2(_regA, _dAv2, _in_id, _in_off)    LOAD_dAv2_SIZE%d(_regA, _dAv2, _in_id, _in_off)\n" % self.dAvn_size)
            f.write("#define LOAD_dBv2(_regB, _dBv2, _dBv2_off)          LOAD_dBv2_SIZE%d(_regB, _dBv2, _dBv2_off)\n\n" % self.dBvn_size)

            f.write("#define MMA_INSTS(_C, _A, _B)                       MMA_INST_2INT_%dx%d(_C, _A, _B)\n\n" % (self.dAvn_size / 2, self.dBvn_size))
        elif self.s_size == 32:
            f.write("#include \"idxn/common/dmem_i4_macros.h\"\n\n")
            f.write("#include \"idxn/common/hmma_i4_macros.h\"\n\n")

            f.write("#define LOAD_dAv4(_regA, _dAv4, _in_id, _in_off)    LOAD_dAv4_SIZE%d(_regA, _dAv4, _in_id, _in_off)\n" % self.dAvn_size)
            f.write("#define LOAD_dBv4(_regB, _dBv4, _dBv4_off)          LOAD_dBv4_SIZE%d(_regB, _dBv4, _dBv4_off)\n\n" % self.dBvn_size)

            f.write("#define MMA_INSTS(_C, _A, _B)                       MMA_INST_4INT_%dx%d(_C, _A, _B)\n\n" % (self.dAvn_size / 2, self.dBvn_size))

        f.write("#include \"idxn/common/output_macros.h\"\n\n")

        f.write("#include \"idxn/common/main_body.h\"\n\n")

        f.write("#include \"idxn/common/uni_undefs.h\"\n\n")

class IdxSourceFile:
    def __init__(self, path):
        self.path = path
        self.fname = "idxn_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include  \"idxn/idxn_kernels.h\"\n\n")

        self.f.write("#define ENABLE_FUSE\n\n")

    def AppendKernel(self, fname):
        self.f.write("#include \"idxn/%s\"\n" % fname)

    def Close(self):
        self.f.close()

class IdxHeaderFile:
    def __init__(self, path):
        self.path = path
        self.fname = "idxn_kernels.h"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#ifndef __PPLCUDA_IDXN_KERNELS_H__\n")
        self.f.write("#define __PPLCUDA_IDXN_KERNELS_H__\n")

        self.f.write("\n\n#include \"kernel_type.h\"\n\n")

    def AppendKernel(self, kname):
        self.f.write("__global__ idx_kernel_t %s;\n" % kname)

    def Close(self):
        self.f.write("\n\n#endif\n")
        self.f.close()

class InitFile:
    def __init__(self, path):
        self.path = path
        self.fname = "init_idxn_kernels.cu"

        self.f = open(os.path.join(self.path, self.fname), "w")

        self.f.write("#include \"conv_common.h\"\n\n")

        self.f.write("#include \"idxn/idxn_kernels.h\"\n\n")

        self.f.write("void InitializeIdxnConvKernelContainer(std::vector<kernel_info_t> & kernel_container)\n{\n")

    def AppendKernel(self, s_size, kname):
        if s_size == 8:
            self.f.write("\tADD_KERNEL(CONV_IDXN_C2, \"%s\", NULL, NULL, &%s);\n" % (kname, kname))
        elif s_size == 16:
            self.f.write("\tADD_KERNEL(CONV_IDXN_C4, \"%s\", NULL, NULL, &%s);\n" % (kname, kname))
        elif s_size == 32:
            self.f.write("\tADD_KERNEL(CONV_IDXN_C32, \"%s\", NULL, NULL, &%s);\n" % (kname, kname))
        else:
            exit(1)

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
    idx_header_file = IdxHeaderFile(parent_path)
    idx_source_file = IdxSourceFile(parent_path)

    init_file = InitFile(parent_path)

    path = parent_path + '/kernels'

    if not os.path.exists(path):
        os.makedirs(path)

    for s_size in [8, 16, 32]:
        for k_num in [1, 2]:
            for warp_y in [16, 32, 64]:
                for warp_x in [8, 16, 32]:
                    for cta_y_num in [1, 2, 4]:
                        for cta_x_num in [1, 2, 4]:
                            if cta_y_num == 4 and cta_x_num == 4:
                                continue

                            kernel = KernelInfo(parent_path, s_size, k_num, cta_y_num, cta_x_num, warp_y, warp_x)

                            kernel.GenKernel()

                            idx_header_file.AppendKernel(kernel.kname)
                            idx_source_file.AppendKernel(kernel.fname)

                            init_file.AppendKernel(s_size, kernel.kname)
    idx_header_file.Close()
    idx_source_file.Close()

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
