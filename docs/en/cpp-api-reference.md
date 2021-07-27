This section introduces some public classes and functions of `PPLNN`.

## Engine

Defined in [include/ppl/nn/engines/engine.h](../../include/ppl/nn/engines/engine.h).

An `Engine` is a collection of op implementations running on specified devices such as CPU or Nvidia GPU.

#### Functions

```c++
ppl::common::RetCode Configure(uint32_t option, ...);
```

Sets various options for this engine. Parameters vary depending on the first parameter `option`.

## X86EngineFactory

A built-in engine factory thas is used to create engines running on x86-compatible CPUs.

#### Functions

```c++
Engine* X86EngineFactory::Create();
```

Creates a x86 engine instance.

## CudaEngineFactory

A built-in engine factory thas is used to create engines running on NVIDIA GPUs.

#### Functions

```c++
Engine* CudaEngineFactory::Create(const CudaEngineOptions& options);
```

Creates a CUDA engine instance with the given `options`.

## OnnxRuntimeBuilderFactory

Defined in [include/ppl/nn/models/onnx/onnx_runtime_builder_factory.h](../../include/ppl/nn/models/onnx/onnx_runtime_builder_factory.h).

Used to create an `OnnxRuntimeBuilder`.

#### Functions

```c++
OnnxRuntimeBuilder* Create(const char* model_file,
                           std::vector<std::unique_ptr<Engine>>&&);
```

Creates an `OnnxRuntimeBuilder` instance from an ONNX model file. The first parameter is the model file path, the second is engines that may be used to evaluate the compute graph.

```c++
OnnxRuntimeBuilder* Create(const char* model_buf, uint64_t buf_len,
                           std::vector<std::unique_ptr<Engine>>&&);
```

Creates an `OnnxRuntimeBuilder` instance from an ONNX buffer.

## OnnxRuntimeBuilder

Defined in [include/ppl/nn/models/onnx/onnx_runtime_builder.h](../../include/ppl/nn/models/onnx/onnx_runtime_builder.h).

`OnnxRuntimeBuilder` is used to create `Runtime` instances. It contains read-only data that a `Runtime` needs.

#### Functions

```c++
Runtime* CreateRuntime(const RuntimeOptions&);
```

Creates a `Runtime` instance which is used to evaluate a compute graph. The parameter `RuntimeOptions` is defined in [include/ppl/nn/runtime/runtime_options.h](../../include/ppl/nn/runtime/runtime_options.h).

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

Gets the number of input of the associated graph.

```c++
Tensor* GetInputTensor(uint32_t idx) const;
```

Gets the input tensor at position `idx`. Note that `idx` should be less than the number of inputs.

```c++
ppl::common::RetCode Run();
```

Runs the model with given inputs. Input data MUST be filled via the returned value of `GetInputTensor()` before calling this function.

```c++
ppl::common::RetCode Sync();
```

Blocks current CPU thread until all operations are finished. Note that this function MUST be called before getting outputs or profiling statistics, in case some engine may run asynchronously.


```c++
uint32_t GetOutputCount() const;
```

Gets the number of outputs of the associated graph.


```c++
Tensor* GetOutputTensor(uint32_t idx) const;
```

Gets the output tensor at position `idx`. Note that `idx` should be less than the number of outputs.

```c++
ppl::common::RetCode GetProfilingStatistics(ProfilingStatistics*) const;
```

Gets profiling statistics of each kernel. Note that this function is available if `PPLNN_ENABLE_KERNEL_PROFILING` is enable.

## Tensor

Defined in [include/ppl/nn/runtime/tensor.h](../../include/ppl/nn/runtime/tensor.h).

This structure represents the input/output data.

#### Functions

```c++
ppl::common::TensorShape& GetShape();
const ppl::common::TensorShape& GetShape() const;
```

Gets the shape of this tensor.

```c++
ppl::common::RetCode ReallocBuffer();
```

Reallocates a buffer according to its shape.

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

Gets the logger for pplnn internal logging. Default is a `StdioLogger` that prints logs to stdout/stderr.

```c++
void Logger::SetLogLevel(uint32_t);
uint32_t Logger::GetLogLevel() const;
```

Sets/gets the level for logging. Any log level that is less than `Logger::GetLogLevel()` will not be logged.


All API references(in html format) can be generated by running `doxygen docs/Doxyfile`.
