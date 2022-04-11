This section describes the python APIs of `PPLNN`. Refer to [pplnn.py](../../tools/pplnn.py) for usage examples and [py_pplnn.cc](../../python/py_pplnn.cc) for exported symbols.

## Common APIs in `pyppl.nn`

### TensorShape

```python
dims = TensorShape::GetDims()
```

Returns an array of dimensions.

```python
TensorShape::SetDims(dims)
```

Sets dims of the tensor.

```python
data_type = TensorShape::GetDataType()
```

Returns the data type of elements in tensor. Data types are defined in `pyppl.common`.

```python
data_format = TensorShape::GetDataFormat()
```

Returns the data format of tensor. Data formats are defined in `pyppl.common`.

```python
is_scalar = TensorShape::IsScalar()
```

Tells whether a tensor is a scalar.

### Tensor

```python
name_str = Tensor::GetName()
```

Returns the tensor's name.

```python
tensor_shape = Tensor::GetShape()
```

Returns a `TensorShape` info of the tensor.

```python
ret_code = Tensor::ConvertFromHost(numpy_ndarray)
```

Copies NDARRAY data to the tensor from an `ndarray` object. `ret_code` is an instance of `RetCode` defined in `pyppl.common`.

```python
tensor_data = Tensor::ConvertToHost(data_type=pplcommon.DATATYPE_UNKNOWN, data_format=pplcommon.DATAFORMAT_NDARRAY)
```

Copies tensor's data to host. If `data_type` or `data_format` is unknown(by setting them to `DATATYPE_UNKNOWN` and `DATAFORMAT_UNKNOWN` respectively), data type or format is unchanged. Then we can use `numpy.array` to create an `ndarray` instance using `numpy_ndarray = numpy.array(tensor_data, copy=False)`.

```python
dev_ctx = Tensor::GetDeviceContext()
```

Returns context of the underlying `Device`.

```python
addr = Tensor::GetBufferPtr()
```

Returns the underlying buffer ptr as an integer.

```python
Tensor::SetBfferPtr(addr)
```

Sets the tensor buffer area to `addr` which is an integer and can be casted to `void*`. Note that `addr` can be read/written by internal `Device` class.

### OnnxRuntimeBuilderFactory

```python
runtime_builder = OnnxRuntimeBuilderFactory.Create()
```

creates an `OnnxRuntimeBuilder` instance.

### OnnxRuntimeBuilder

```python
status = runtime_builder.InitFromFile(onnx_model_file, engines)
```

Initializes an `OnnxRuntimeBuilder` instance from an ONNX model. `engines` is a list of `Engine` instances that may be used to evaluate the model.

```python
status = runtime_builder.Preprocess()
```

does some preparations before creating `Runtime` instances.

```python
runtime = runtime_builder.CreateRuntime()
```

Creates a `Runtime` instance for inferencing.

### Runtime

```python
input_count = Runtime::GetInputCount()
```

Returns the number of model inputs.

```python
input_tensor = Runtime::GetInputTensor(idx)
```

Returns the input tensor in position `idx`, which is in range [0, input_count).

```python
ret_code = Runtime::Run()
```

Evaluates the model. `ret_code` is an instance of `RetCode` defined in `pyppl.common`.

```python
output_count = Runtime::GetOutputCount()
```

Returns the number of model outputs.

```python
output_tensor = Runtime::GetOutputTensor(idx)
```

Returns the output tensor in position `idx`, which is in range [0, output_count).

```python
dev_count = Runtime::GetDeviceContextCount()
```

Returns the number of `DeviceContext` used by this `Runtime` instance.

```python
dev_ctx = Runtime::GetDeviceContext(idx)
```

Returns the `DeviceContext` at position `idx`. Note that `idx` should be less than `GetDeviceContextCount()`.

## Device Specific APIs in `pyppl.nn`

### X86

#### X86EngineFactory

```python
x86_options = X86EngineOptions()
x86_engine = X86EngineFactory::Create(x86_options)
```

Creates an `Engine` instance running on x86-64 compatiable CPUs.

```python
ret_code = x86_engine.Configure(option, <optional parameters>)
```

Configures `x86_engine`. Refer to [x86_options.h](../../include/ppl/nn/engines/x86/x86_options.h) for available options.

### CUDA

#### CudaEngineOptions

Refer to [cuda_engine_options.h](../../include/ppl/nn/engines/cuda/cuda_engine_options.h) for more details.

#### CudaEngineFactory

```python
cuda_options = CudaEngineOptions()
cuda_engine = CudaEngineFactory::Create(cuda_options)
```

Creates an `Engine` instance running on NVIDIA GPUs.

```python
ret_code = cuda_engine.Configure(option, <optional parameters>)
```

Configures `cuda_engine`. Refer to [cuda_options.h](../../include/ppl/nn/engines/cuda/cuda_options.h) for available options(some options are not exported yet).

## Other Utilities

```python
version_str = pypplnn.GetVersionString()
```

Returns the version string of current version.

```python
msg_str = pyppl.common.GetRetCodeStr(ret_code)
```

Returns a human-readable message of `ret_code`.

```python
pyppl.common.SetLoggingLevel(log_level)
log_level = pyppl.common.GetLoggingLevel()
```

Sets and gets the current logging level respectively. Logging levels are defined in `pyppl.common`.
