This section introduces some public classes and functions of `PPLNN`.

For brevity, we assume that `using namespace ppl::nn;` is always used in the following code snippets.

## Engine and Ops

Defined in [include/ppl/nn/engines/engine.h](../../include/ppl/nn/engines/engine.h).

An `Engine` is a collection of op implementations running on specified devices such as CPU or Nvidia GPU.

#### Functions

```c++
ppl::common::RetCode Configure(uint32_t option, ...);
```

Sets various options for this engine. Parameters vary depending on the first parameter `option`.

## x86::EngineFactory

A built-in engine factory that is used to create engines running on x86-compatible CPUs.

If you want to use built-in op implementations, you should call `x86::RegisterBuiltinOps()` manually.

#### Functions

```c++
Engine* x86::EngineFactory::Create(const x86::EngineOptions& options);
```

Creates an X86 engine instance with the specified options.

## cuda::EngineFactory

A built-in engine factory that is used to create engines running on NVIDIA GPUs.

If you want to use built-in op implementations, you should call `RegisterBuiltinOps()` manually.

#### Functions

```c++
Engine* cuda::EngineFactory::Create(const cuda::EngineOptions& options);
```

Creates a CUDA engine instance with the specified options.

## arm::EngineFactory

A built-in engine factory that is used to create engines running on arm aarch64 CPUs.

If you want to use built-in op implementations, you should call `arm::RegisterBuiltinOps()` manually.

#### Functions

```c++
Engine* arm::EngineFactory::Create(const arm::EngineOptions& options);
```

Creates a ARM engine instance with the specified options.

## riscv::EngineFactory

A built-in engine factory that is used to create engines running on riscv64 CPUs.

If you want to use built-in op implementations, you should call `riscv::RegisterBuiltinOps()` manually.

#### Functions

```c++
Engine* riscv::EngineFactory::Create(const riscv::EngineOptions& options);
```

Creates an RISCV engine instance with the specified options.

## onnx::RuntimeBuilderFactory

Defined in [include/ppl/nn/models/onnx/runtime_builder_factory.h](../../include/ppl/nn/models/onnx/runtime_builder_factory.h).

#### Functions

```c++
onnx::RuntimeBuilder* Create();
```

Creates an `onnx::RuntimeBuilder` instance.

## onnx::RuntimeBuilder

Defined in [include/ppl/nn/models/onnx/runtime_builder.h](../../include/ppl/nn/models/onnx/runtime_builder.h).

`onnx::RuntimeBuilder` is used to create `Runtime` instances.

#### Functions

```c++
ppl::common::RetCode LoadModel(const char* model_file);
ppl::common::RetCode LoadModel(const char* model_buf, uint64_t buf_len, const char* model_file_dir = nullptr);
```

Initializes an `onnx::RuntimeBuilder` instance from an ONNX model file or buffer. `model_file_dir` is used to parse external data and can be `nullptr` if there is no external data.

```c++
struct Resources final {
    /** `engines` are used to evaluate the compute graph. Note that callers should guarantee that engines are valid during inferencing. */
    Engine** engines;
    uint32_t engine_num;
};

ppl::common::RetCode SetResources(const Resources&);
```

Sets the resources needed for preprocessing and evaluating models.

```c++
ppl::common::RetCode Preprocess();
```

prepare for creating `Runtime` instances.


```c++
Runtime* CreateRuntime();
```

Creates a `Runtime` instance which is used to evaluate a compute graph.

## pmx::RuntimeBuilderFactory

Defined in [include/ppl/nn/models/pmx/runtime_builder_factory.h](../../include/ppl/nn/models/pmx/runtime_builder_factory.h).

#### Functions

```c++
pmx::RuntimeBuilder* Create();
```

Creates an `pmx::RuntimeBuilder` instance.

## pmx::RuntimeBuilder

Defined in [include/ppl/nn/models/pmx/runtime_builder.h](../../include/ppl/nn/models/pmx/runtime_builder.h).

`pmx::RuntimeBuilder` is used to create `Runtime` instances.

#### Functions

```c++
ppl::common::RetCode LoadModel(const char* model_file);
ppl::common::RetCode LoadModel(const char* model_buf, uint64_t buf_len);
```

Initializes an `pmx::RuntimeBuilder` instance from an PMX model file or buffer.

```c++
struct Resources final {
    /** `engines` are used to evaluate the compute graph. Note that callers should guarantee that engines are valid during inferencing. */
    Engine** engines;
    uint32_t engine_num;
};

ppl::common::RetCode SetResources(const Resources&);
```

Sets the resources needed for preprocessing and evaluating models.

```c++
ppl::common::RetCode Preprocess();
```

prepare for creating `Runtime` instances.


```c++
Runtime* CreateRuntime();
```

Creates a `Runtime` instance which is used to evaluate a compute graph.

## Runtime

Defined in [include/ppl/nn/runtime/runtime.h](../../include/ppl/nn/runtime/runtime.h).

`Runtime` is the main structure for evaluating a model.

#### Functions

```c++
ppl::common::RetCode Configure(uint32_t option, ...);
```

Sets various runtime options defined in `runtime_options.h`. Parameters vary depending on the first parameter `option`.

```c++
uint32_t GetInputCount() const;
```

Returns the number of input of the associated graph.

```c++
Tensor* GetInputTensor(uint32_t idx) const;
```

Returns the input tensor at position `idx`. Note that `idx` should be less than the number of inputs.

```c++
ppl::common::RetCode Run();
```

Runs the model with given inputs. Input data MUST be filled via the returned value of `GetInputTensor()` before calling this function.

```c++
uint32_t GetOutputCount() const;
```

Returns the number of outputs of the associated graph.


```c++
Tensor* GetOutputTensor(uint32_t idx) const;
```

Returns the output tensor at position `idx`. Note that `idx` should be less than the number of outputs.

```c++
uint32_t GetDeviceContextCount() const;
```

Returns the number of `DeviceContext` used by this `Runtime` instance.

```c++
DeviceContext* GetDeviceContext(uint32_t idx) const;
```

Returns the `DeviceContext` at position `idx`. Note that `idx` should be less than `GetDeviceContextCount()`.

```c++
ppl::common::RetCode GetProfilingStatistics(ProfilingStatistics*) const;
```

Returns profiling statistics of each kernel. Note that this function is available if `PPLNN_ENABLE_KERNEL_PROFILING` is enable.

## Tensor

Defined in [include/ppl/nn/runtime/tensor.h](../../include/ppl/nn/runtime/tensor.h).

This structure represents the input/output data.

#### Functions

```c++
ppl::common::TensorShape* GetShape() const;
```

Returns the shape of this tensor.

```c++
ppl::common::RetCode ReallocBuffer();
```

Reallocates a buffer according to its shape.

```c++
void FreeBuffer();
```

Frees the data buffer.

```c++
ppl::common::RetCode CopyToHost(void* dst) const;
```

Copies tensor's data to `dst` which points to a host memory. Note that `dst` MUST have enough space.

```c++
ppl::common::RetCode CopyFromHost(const void* src);
```

Copies data to inner buffer from `src`, which points to a host memory. Note that inner buffer MUST be allocated before calling this function.

```c++
ppl::common::RetCode ConvertToHost(void* dst, const ppl::common::TensorShape& dst_desc) const;
```

Converts tensor's data to `dst` with the shape `dst_desc`.

```c++
ppl::common::RetCode ConvertFromHost(const void* src, const ppl::common::TensorShape& src_desc);
```

Converts data to inner buffer from `src` with the shape `src_desc`. Note that inner buffer MUST be allocated before calling this function.

```c++
DeviceContext* GetDeviceContext() const;
```

Returns context of the underlying `Device`.

```c++
void SetBufferPtr(void* buf);
```

Sets the underlying buffer ptr. Note that `buf` can be read/written by the internal `Device` class.

```c++
void* GetBufferPtr() const;
```

Returns the underlying buffer ptr.

## TensorShape

Defined in [include/ppl/nn/common/tensor_shape.h](../../include/ppl/nn/common/tensor_shape.h).

## Logger

Defined in [include/ppl/nn/common/logger.h](../../include/ppl/nn/common/logger.h).

```c++
void SetCurrentLogger(Logger*);
```

Sets global logger for pplnn internal logging.

```c++
Logger* GetCurrentLogger();
```

Returns the logger for pplnn internal logging. Default is a `StdioLogger` that prints logs to stdout/stderr.

```c++
void Logger::SetLogLevel(uint32_t);
uint32_t Logger::GetLogLevel() const;
```

Sets/gets the level for logging. Any log level that is less than `Logger::GetLogLevel()` will not be logged.


All API references(in html format) can be generated by running `doxygen docs/Doxyfile`.
