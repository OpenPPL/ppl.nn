# -*- coding: utf-8 -*-

"""
Init all include file into gene_header.cc
"""

import os
import sys

def write_head(file):
    template = """// Licensed to the Apache Software Foundation (ASF) under one
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
#include "gene_header.h"
#include <fstream>
#include <sstream>
std::string GeneHeader::Find(const std::string& path) {
    auto header_ref = header_code_.find(path);
    if (header_ref != header_code_.end()) {
        return header_ref->second;
    }
    return "";
}
GeneHeader::GeneHeader() {
    std::string code_str;
"""
    file.write(template)

def write_tail(file):
    template = """}
"""
    file.write(template)

def init_include_file(file, path, name):
    with open(path + name, "r", encoding='UTF-8') as header_file:
        code = header_file.read()
        code = code.replace('\\', '\\\\')
        code = code.replace('\"', '\\\"')
        code = code.replace('\n', '\\n";\n    code_str += \"')
        template = """    code_str = "{code}";
    header_code_.emplace("{name}", code_str);
"""
        file.write(template.format(name = name,
                                   code = code))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    sourse_pwd = sys.argv[1]
    tail_pwd = sys.argv[2]

    with open(tail_pwd + "/gene_header.cc", "w+", encoding='UTF-8') as f:
        write_head(f)

        init_include_file(f, sourse_pwd, "/2spk/fp16/async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/const_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/hmma_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/ldsm_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/main_body16816.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/main_body1688.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/output_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/reduce_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/smem_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/uni_undefs.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/f1/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/f3/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/fn/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/fs/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/f1/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/f3/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/fn/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/fs/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/f1/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/f3/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/fn/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fp16/fs/dmem_reg_macros.h")

        init_include_file(f, sourse_pwd, "/2spk/int8/async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/const_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/imma16816_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/imma16832_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/imma8816_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/ldsm_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/main_body16816.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/main_body16832.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/main_body8816.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/output_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/reduce_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/smem_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/uni_undefs.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/f1/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/f3/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/fn/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/fs/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/f1/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/f3/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/fn/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/fs/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/f1/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/f3/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/fn/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/int8/fs/dmem_reg_macros.h")

        init_include_file(f, sourse_pwd, "/swzl/fp16/async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/const_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/hmma_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/ldsm_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/main_body16816.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/main_body1688.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/output_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/reduce_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/smem_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/uni_undefs.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/f1/bound_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/f3/bound_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/fn/bound_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/f1/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/f3/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/fn/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/f1/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/f3/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/fp16/fn/dmem_reg_macros.h")

        init_include_file(f, sourse_pwd, "/swzl/int8/async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/const_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/imma16816_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/imma16832_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/imma8816_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/ldsm_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/main_body16816.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/main_body16832.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/main_body8816.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/output_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/reduce_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/smem_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/uni_undefs.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/f1/bound_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/f3/bound_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/fn/bound_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/f1/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/f3/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/fn/dmem_async_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/f1/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/f3/dmem_reg_macros.h")
        init_include_file(f, sourse_pwd, "/swzl/int8/fn/dmem_reg_macros.h")

        init_include_file(f, sourse_pwd, "/idxn/fp16/const_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/dmem_i1_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/dmem_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/dmem_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/hmma16816_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/hmma16816_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/hmma1688_i1_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/hmma1688_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/hmma1688_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/main_body.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/output_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/fp16/uni_undefs.h")

        init_include_file(f, sourse_pwd, "/idxn/int8/const_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/dmem_i1_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/dmem_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/dmem_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma16816_i1_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma16816_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma16816_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma16816_output_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma16832_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma16832_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma8816_i1_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma8816_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma8816_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/imma8816_output_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/main_body.h")
        init_include_file(f, sourse_pwd, "/idxn/int8/uni_undefs.h")

        write_tail(f)


if __name__ == "__main__":
    main()
