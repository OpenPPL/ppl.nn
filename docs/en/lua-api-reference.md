This section describes the Lua APIs of `PPLNN`. Refer to [pplnn.lua](../../tools/pplnn.lua) for usage examples and [lua_pplnn.cc](../../lua/lua_pplnn.cc) for exported symbols.

## Common APIs in `luappl.nn`

### TensorShape

```lua
dims = TensorShape:GetDims()
```

Returns an array of dimensions.

```lua
TensorShape:SetDims(dims)
```

Sets dims of the tensor.

```lua
data_type = TensorShape:GetDataType()
```

Returns the data type of elements in tensor. Data types are defined in `luappl.common`.

```lua
data_format = TensorShape:GetDataFormat()
```

Returns the data format of tensor. Data formats are defined in `luappl.common`.

```lua
is_scalar = TensorShape:IsScalar()
```

Tells whether a tensor is a scalar.

### Tensor

```lua
name_str = Tensor:GetName()
```

Returns the tensor's name.

```lua
tensor_shape = Tensor:GetShape()
```

Returns a `TensorShape` object of the tensor.

```lua
ret_code = Tensor:ConvertFromHost(buffer, dims, data_type)
```

Copies data to the tensor from buffer shaped as `dims` and `data_type`. `ret_code` is an instance of `RetCode` defined in `luappl.common`. `data_type` is defined in `luappl.common`.

```lua
tensor_data = Tensor:ConvertToHost()
```

Copies tensor's data to host in NDARRAY format. Shape and data type can be retrieved by `Tensor:GetShape():GetDims()` and `Tensor:GetShape():GetDataType()` respectively.

### Engine

```lua
name_str = Engine:GetName()
```

Returns engine's name.

### onnx.RuntimeBuilderFactory

```lua
runtime_builder = onnx.RuntimeBuilderFactory:Create()
```

Creates an `onnx.RuntimeBuilder` instance.

### onnx.RuntimeBuilder

```lua
status = runtime_builder:LoadModelFromFile(onnx_model_file)
```

loads an ONNX model from the specified file.

```lua
resources = RuntimeBuilderResources()
resources.engines = engines
status = runtime_builder:SetResources(resources)
```

where `engines` is a list of `Engine` instances that are used to preprocess and evaluate the model.

```lua
status = runtime_builder:Preprocess()
```

does some preparations before creating `Runtime` instances.

```lua
runtime = RuntimeBuilder:CreateRuntime()
```

creates a `Runtime` instance for inferencing.

### Runtime

```lua
input_count = Runtime:GetInputCount()
```

Returns the number of model inputs.

```lua
input_tensor = Runtime:GetInputTensor(idx)
```

Returns the input tensor in position `idx`, which is in range [0, input_count).

```lua
ret_code = Runtime::Run()
```

Evaluates the model. `ret_code` is an instance of `RetCode` defined in `luappl.common`.

```lua
output_count = Runtime:GetOutputCount()
```

Returns the number of model outputs.

```lua
output_tensor = Runtime:GetOutputTensor(idx)
```

Returns the output tensor in position `idx`, which is in range [0, output_count).

## Device Specific APIs in `luappl.nn`

### x86

#### EngineOptions

Refer to [engine_options.h](../../include/ppl/nn/engines/x86/engine_options.h) for more details.

#### EngineFactory

```lua
x86_options = x86.EngineOptions()
x86_engine = x86.EngineFactory:Create(x86_options)
```

Creates an `Engine` instance running on x86-64 compatiable CPUs.

### CUDA

#### EngineOptions

Refer to [engine_options.h](../../include/ppl/nn/engines/cuda/engine_options.h) for more details.

#### EngineFactory

```lua
cuda_options = cuda.EngineOptions()
cuda_engine = cuda.EngineFactory:Create(cuda_options)
```

Creates an `Engine` instance running on NVIDIA GPUs.

## Other Utilities

```lua
version_str = luappl.nn.GetVersionString()
```

Returns the version string of current version.

```lua
msg_str = luappl.common.GetRetCodeStr(ret_code)
```

Returns a human-readable message of `ret_code`.

```lua
luappl.common.SetLoggingLevel(log_level)
log_level = luappl.common.GetLoggingLevel()
```

Sets and gets the current logging level respectively. Logging levels are defined in `luappl.common`.
