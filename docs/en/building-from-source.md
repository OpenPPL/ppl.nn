### Prerequisites

* Linux running on x86 compatible CPUs
* GCC >= 4.9 or LLVM/Clang >= 6.0
* [CMake](https://cmake.org/download/) >= 3.13
* [Git](https://git-scm.com/downloads) >= 2.7.0
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) >= 10.2 (for CUDA version)
* [Python](https://www.python.org/downloads/) >= 3 (for CUDA version)

### Download the Source Code

```bash
git clone https://github.com/openppl-public/ppl.nn.git
```

### Build X86-64 Engine

```bash
./build.sh
```

Headers and libraries are installed in `pplnn-build/install`.

If you want to enable openmp, please specify `HPCC_USE_OPENMP` as following:

```bash
./build.sh -DHPCC_USE_OPENMP=ON
```

If you are building on MacOS (Darwin), install `libomp` by [homebrew](https://brew.sh/) first:
```bash
brew install libomp
```

### Enable CUDA Engine

X86-64 engine is enabled by default.

```bash
./build.sh -DHPCC_USE_CUDA=ON
```

Headers and libraries are installed in `pplnn-build/install`.

If you want to use specified CUDA toolkit version, please specify `CUDA_TOOLKIT_ROOT_DIR` as following:

```bash
./build.sh -DHPCC_USE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda-toolkit-root-dir
```

### Test

There is a test tool named `pplnn` in `tools/pplnn.cc`. You can run `pplnn` using the following command:

```bash
./pplnn-build/tools/pplnn --onnx-model tests/testdata/conv.onnx
```

NOTE: if CUDA engine is enabled, `pplnn` uses CUDA only.
