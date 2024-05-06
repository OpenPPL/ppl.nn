#!/bin/bash

workdir=`pwd`

if [ -z "$PPL_BUILD_THREAD_NUM" ]; then
    PPL_BUILD_THREAD_NUM=32
    echo -e "env 'PPL_BUILD_THREAD_NUM' is not set. use PPL_BUILD_THREAD_NUM=${PPL_BUILD_THREAD_NUM} by default."
fi

build_type='Release'
options="-DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_INSTALL_PREFIX=install $*"

pplnn_build_dir="${workdir}/pplnn-build"
mkdir ${pplnn_build_dir}
cd ${pplnn_build_dir}
cmd="cmake $options .. && cmake --build . -j ${PPL_BUILD_THREAD_NUM} --config ${build_type} && cmake --build . --target install -j ${PPL_BUILD_THREAD_NUM} --config ${build_type}"
echo "cmd -> $cmd"
eval "$cmd"
