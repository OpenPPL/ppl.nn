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
    with open(path + name, "r") as header_file:
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

    with open(tail_pwd + "/gene_header.cc", "w+") as f:
        write_head(f)
        
        init_include_file(f, sourse_pwd, "/2spk/common/const_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/f1/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/f3/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fn/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fs/bound_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/common/ldsm_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/f1/dmem_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/f3/dmem_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fn/dmem_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/fs/dmem_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/common/hmma_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/common/reduce_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/common/smem_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/common/output_macros.h")
        init_include_file(f, sourse_pwd, "/2spk/common/main_body.h")
        init_include_file(f, sourse_pwd, "/2spk/common/uni_undefs.h")
        init_include_file(f, sourse_pwd, "/idxn/common/const_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/dmem_i1_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/hmma_i1_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/dmem_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/hmma_i2_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/dmem_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/hmma_i4_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/output_macros.h")
        init_include_file(f, sourse_pwd, "/idxn/common/main_body.h")
        init_include_file(f, sourse_pwd, "/idxn/common/uni_undefs.h")
         
        write_tail(f)
        

if __name__ == "__main__":
    main()