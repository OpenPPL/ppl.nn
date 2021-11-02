#!/bin/bash

workdir=`pwd`

if [[ `uname` == "Linux" ]]; then
    processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`
elif [[ `uname` == "Darwin" ]]; then
    processor_num=`sysctl machdep.cpu | grep machdep.cpu.core_count | cut -d " " -f 2`
else
    processor_num=1
fi

options="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install $*"

pplnn_build_dir="${workdir}/pplnn-build"
mkdir ${pplnn_build_dir}
cd ${pplnn_build_dir}
cmd="cmake $options .. && cmake --build . -j ${processor_num} --config Release && cmake --build . --target install -j ${processor_num} --config Release"
echo "cmd -> $cmd"
eval "$cmd"
