### Prerequisites

* Linux or Windows running on x86_64 or arm64 CPUs
* GCC >= 4.9 or LLVM/Clang >= 6.0, or Visual Studio >= 2015
* [CMake](https://cmake.org/download/) >= 3.14
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 9.0 (for CUDA)
* [Python](https://www.python.org/downloads/) >= 3.5 (for CUDA and Python API support)
* [Lua](https://www.lua.org/download.html) >= 5.2.0 (optional, for Lua API support)

### Cloning Source Code

```bash
git clone https://github.com/openppl-public/ppl.nn.git
```

### Building X86_64 Engine

#### Linux

```bash
./build.sh -DPPLNN_USE_X86_64=ON
```

Headers and libraries are installed in `pplnn-build/install`.

If you want to enable openmp, please specify `PPLNN_USE_OPENMP` as following:

```bash
./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_USE_OPENMP=ON
```

#### Windows

Using vs2015 for example:

```
build.bat -G "Visual Studio 14 2015 Win64" -DPPLNN_USE_X86_64=ON
```

Headers and libraries are installed in `pplnn-build/install`.

### Building CUDA Engine

#### Linux

```bash
./build.sh -DPPLNN_USE_CUDA=ON
```

Note that if you want to build X86 engine along with CUDA engine, you should specify `-DPPLNN_USE_X86_64=ON` explicitly like this:

```bash
./build.sh -DPPLNN_USE_X86_64=ON -DPPLNN_USE_CUDA=ON
```

Headers and libraries are installed in `pplnn-build/install`.

If you want to use specified CUDA toolkit version, please specify `CUDA_TOOLKIT_ROOT_DIR` as following:

```bash
./build.sh -DPPLNN_USE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda-toolkit-root-dir
```

#### Cross Compiling for Arm on X86

Using the following command:

```bash
CUDA_TOOLKIT_ROOT=/path/to/cuda/toolkit/root/dir ./build.sh -DPPLNN_USE_CUDA=ON -DPPLNN_TOOLCHAIN_DIR=/path/to/arm/toolchain/dir -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake
```

Note that the `CUDA_TOOLKIT_ROOT` environment variable is required.

You can also specify `CUDA_TOOLKIT_ROOT_DIR` without setting `CUDA_TOOLKIT_ROOT`, which will be set to `CUDA_TOOLKIT_ROOT_DIR` by ppl.nn:

```bash
./build.sh -DPPLNN_USE_CUDA=ON -DPPLNN_TOOLCHAIN_DIR=/path/to/arm/toolchain/dir -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit/root/dir
```

#### Windows

Using vs2015 for example:

```
build.bat -G "Visual Studio 14 2015 Win64" -DPPLNN_USE_CUDA=ON
```

Headers and libraries are installed in `pplnn-build/install`.

#### Other useful options

We use runtime-compiling version by default. If you want to use static version (build all kernels in advance), please specify `PPLNN_ENABLE_CUDA_JIT` as following:

```bash
./build.sh -DPPLNN_USE_CUDA=ON -DPPLNN_ENABLE_CUDA_JIT=OFF
```

If you want to run debug model, please specify `CMAKE_BUILD_TYPE` as following:

```bash
./build.sh -DPPLNN_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
```

If you want to profile running time for each kernel, please specify `PPLNN_ENABLE_KERNEL_PROFILING` as following and add arg `--enable-profiling` during executing pplnn.

```bash
./build.sh -DPPLNN_USE_CUDA=ON -DPPLNN_ENABLE_KERNEL_PROFILING=ON
```

### Building RISCV Engine

#### AllWinner D1

You need to download c906 toolchain package from [https://occ.t-head.cn/community/download?id=3913221581316624384](https://occ.t-head.cn/community/download?id=3913221581316624384).
``` bash
tar -xf riscv64-linux-x86_64-20210512.tar.gz
export RISCV_ROOT_PATH=/path/to/riscv64-linux-x86_64-20210512
```

Build pplnn:
```bash
./build.sh -DPPLNN_TOOLCHAIN_DIR=$RISCV_ROOT_PATH -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake -DPPLNN_USE_RISCV64=ON -DPPLNN_ENABLE_KERNEL_PROFILING=ON -DPPLNN_ENABLE_PYTHON_API=OFF -DPPLNN_ENABLE_LUA_API=OFF -DCMAKE_INSTALL_PREFIX=pplnn-build/install
```

Headers and libraries are installed in `pplnn-build/install`.

### Building ARM Engine

#### Linux

```bash
./build.sh -DPPLNN_USE_AARCH64=ON
```

Headers and libraries are installed in `pplnn-build/install`.

If you want to enable openmp, please specify `PPLNN_USE_OPENMP` as following:

```bash
./build.sh -DPPLNN_USE_AARCH64=ON -DPPLNN_USE_OPENMP=ON
```

If you want to enable FP16 inference, please specify `PPLNN_USE_ARMV8_2` (your compiler must have `armv8.2-a` ISA support):

```bash
./build.sh -DPPLNN_USE_AARCH64=ON -DPPLNN_USE_ARMV8_2=ON
```

If your system has multiple NUMA nodes, it is recommended to build with `PPLNN_USE_NUMA` (please make sure `libnuma` has been installed in your system):

```bash
./build.sh -DPPLNN_USE_AARCH64=ON -DPPLNN_USE_NUMA=ON
```

If you want to run on mobile platforms, please use the Android NDK package:

```bash
./build.sh -DPPLNN_USE_AARCH64=ON -DANDROID_PLATFORM=android-22 -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DCMAKE_TOOLCHAIN_FILE=<path_to_android_ndk_package>/android-ndk-r22b/build/cmake/android.toolchain.cmake
```

### Buliding Python API support

add `-DPPLNN_ENABLE_PYTHON_API=ON` to the build command if you want to use `PPLNN` in python:

```bash
./build.sh -DPPLNN_ENABLE_PYTHON_API=ON
```

If you want to use a specified version of python, you can pass `PYTHON3_INCLUDE_DIRS` to `build.sh`:

```bash
./build.sh -DPPLNN_ENABLE_PYTHON_API=ON -DPYTHON3_INCLUDE_DIRS=/path/to/your/python/include/dir [other options]
```

Run the python demo with the following command:

```bash
PYTHONPATH=./pplnn-build/install/lib python3 ./tools/pplnn.py [--use-x86 | --use-cuda] --onnx-model tests/testdata/conv.onnx
```

or use both engines:

```bash
cd ppl.nn
PYTHONPATH=./pplnn-build/install/lib python3 ./tools/pplnn.py --use-x86 --use-cuda --onnx-model tests/testdata/conv.onnx
```

There is a python packaging configuration in [python/package](../../python/package). You can build a `.whl` package:

```bash
./build.sh
```

and then install this package with `pip`:

```bash
cd /tmp/pyppl-package/dist
pip3 install pyppl*.whl
```

After installation, you can use `from pyppl import nn` directly without setting the `PYTHONPATH` env.

### Buliding Lua API support

add `-DPPLNN_ENABLE_LUA_API=ON` to the build command if you want to use `PPLNN` in lua:

```bash
./build.sh -DPPLNN_ENABLE_LUA_API=ON
```

If you want to use a specified version of lua, you can pass `LUA_SRC_DIR` to `build.sh`:

```bash
./build.sh -DPPLNN_ENABLE_LUA_API=ON -DLUA_SRC_DIR=/path/to/lua/src [other options]
```

or you already have a pre-compiled version, you can pass `LUA_INCLUDE_DIR` and `LUA_LIBRARIES` to `build.sh`:

```bash
./build.sh -DPPLNN_ENABLE_LUA_API=ON -DLUA_INCLUDE_DIR=/path/to/your/lua/include/dir -DLUA_LIBRARIES=/path/to/your/lua/lib [other options]
```

Run the lua demo with the following commands:

```bash
cd ppl.nn
LUAPATH=./pplnn-build/install/lib /path/to/your/lua-interpreter ./tools/pplnn.lua
```

Note that your lua interpreter should be compiled with options `MYCFLAGS="-DLUA_USE_DLOPEN -fPIC" MYLIBS=-ldl` to enable loading .so plugins.

### Testing

There is a test tool named `pplnn` generated from `tools/pplnn.cc`. You can run `pplnn` using the following command:

```bash
./pplnn-build/tools/pplnn [--use-x86 | --use-cuda] --onnx-model tests/testdata/conv.onnx
```
