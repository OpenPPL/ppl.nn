#!/bin/bash

workdir=`pwd`
pplnn_build_dir="${workdir}/pplnn-build"
processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`

options="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${pplnn_build_dir}/install $*"

mkdir ${pplnn_build_dir}
cd ${pplnn_build_dir}
cmd="cmake $options .. && make -j${processor_num} && make install"
echo "cmd -> $cmd"
eval "$cmd"
