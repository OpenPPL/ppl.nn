#!/bin/bash

workdir=`pwd`
x86_64_build_dir="${workdir}/x86-64-build"
cuda_build_dir="${workdir}/cuda-build"
processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`

options='-DCMAKE_BUILD_TYPE=Release'

# --------------------------------------------------------------------------- #

function BuildCuda() {
    mkdir ${cuda_build_dir}
    cd ${cuda_build_dir}
    cmd="cmake $options -DHPCC_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${cuda_build_dir}/install .. && make -j${processor_num} && make install"
    echo "cmd -> $cmd"
    eval "$cmd"
}

function BuildX86_64() {
    mkdir ${x86_64_build_dir}
    cd ${x86_64_build_dir}
    cmd="cmake $options -DHPCC_USE_X86_64=ON -DCMAKE_INSTALL_PREFIX=${x86_64_build_dir}/install .. && make -j${processor_num} && make install"
    echo "cmd -> $cmd"
    eval "$cmd"
}

declare -A engine2func=(
    ["cuda"]=BuildCuda
    ["x86_64"]=BuildX86_64
)

# --------------------------------------------------------------------------- #

function Usage() {
    echo -n "[INFO] usage: $0 [ all"
    for engine in ${!engine2func[@]}; do
        echo -n " | $engine"
    done
    echo "] [cmake options]"
}

if [ $# -lt 1 ]; then
    Usage
    exit 1
fi
engine="$1"

shift
options="$options $*"

if [ "$engine" == "all" ]; then
    for engine in "${!engine2func[@]}"; do
        func=${engine2func[$engine]}
        eval $func
        if [ $? -ne 0 ]; then
            echo "[ERROR] build [$engine] failed." >&2
            exit 1
        fi
    done
else
    func=${engine2func["$engine"]}
    if ! [ -z "$func" ]; then
        eval $func
        if [ $? -ne 0 ]; then
            echo "[ERROR] build [$engine] failed." >&2
            exit 1
        fi
    else
        echo "[ERROR] unknown engine name [$engine]" >&2
        Usage
        exit 1
    fi
fi
