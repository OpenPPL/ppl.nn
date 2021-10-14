This section shows how to use `PPLNN` step by step with an example [api_intro.cc](../../samples/cpp/api/api_intro.cc). Refer to [API Reference](cpp-api-reference.md) for more details.

### Creating engines

In `PPLNN`, an `Engine` is a collection of op implementations running on specified devices such as CPU or NVIDIA GPU. For example, we can use the built-in `X86EngineFactory`:

```c++
Engine* X86EngineFactory::Create();
```

to create an engine which runs on x86-compatible CPUs:

```c++
X86EngineOptions x86_options;
Engine* x86_engine = X86EngineFactory::Create(x86_options);
```

Or use

```c++
CudaEngineOptions cuda_options;
// ... set options
Engine* CudaEngineFactory::Create(cuda_options);
```

to create an engine running on NVIDIA GPUs.

### Creating a RuntimeBuilder

We create a `RuntimeBuilder` with the following function:

```c++
RuntimeBuilder* OnnxRuntimeBuilderFactory::Create(
    const char* model_file, Engine** engines, uint32_t engine_num);
```

where the second parameter `engines` is the `x86_engine` we created:

```c++
vector<unique_ptr<Engine>> engines;
engines.emplace_back(unique_ptr<Engine>(x86_engine));
```

`PPLNN` supports multiple engines running in the same model. For example:

```c++
Engine* x86_engine = X86EngineFactory::Create(X86EngineOptions());
Engine* cuda_engine = CudaEngineFactory::Create(CudaEngineOptions());

vector<unique_ptr<Engine>> engines;
engines.emplace_back(unique_ptr<Engine>(x86_engine));
engines.emplace_back(unique_ptr<Engine>(cuda_engine));
// TODO add other engines

const char* model_file = "/path/to/onnx/model";
// use x86 and cuda engines to run this model
vector<Engine*> engine_ptrs = {x86_engine.get(), cuda_engine.get()};
RuntimeBuilder* builder = OnnxRuntimeBuilderFactory::Create(model_file, engine_ptrs.data(), engine_ptrs.size());
```

`PPLNN` will partition the model and assign different ops to these engines according to configurations.

### Creating a Runtime

We can use

```c++
Runtime* RuntimeBuilder::CreateRuntime();
```

to create a `Runtime`:

```c++
Runtime* runtime = builder->CreateRuntime();
```

### Filling Inputs

We can get graph inputs using the following functions of `Runtime`:

```c++
uint32_t Runtime::GetInputCount() const;
Tensor* Runtime::GetInputTensor(uint32_t idx) const;
```

and fill input data(using random data in this example):

```c++
for (uint32_t c = 0; c < runtime->GetInputCount(); ++c) {
    auto t = runtime->GetInputTensor(c);
    auto& shape = t->GetShape();

    auto nr_element = shape.GetBytesIncludingPadding() / sizeof(float);
    vector<float> buffer(nr_element);

    // fill random input data
    std::default_random_engine eng;
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (uint32_t i = 0; i < nr_element; ++i) {
        buffer[i] = dis(eng);
    }

    auto status = t->ReallocBuffer();
    if (status != RC_SUCCESS) {
        // ......
    }

    // our random data is treated as NDARRAY
    TensorShape src_desc = t->GetShape();
    src_desc.SetDataFormat(DATAFORMAT_NDARRAY);

    // input tensors may require different data format
    status = t->ConvertFromHost((const void*)buffer.data(), src_desc);
    if (status != RC_SUCCESS) {
        // ......
    }
}
```

### Evaluating the Compute Graph

use the `Runtime::Run()`:

```c++
RetCode status = runtime->Run();
```

### Getting Results

Before getting results we must wait for all operations to finish(some engine may run asynchronously):

```c++
RetCode status = runtime->Sync();
```

Then iterate each output:

```c++
for (uint32_t c = 0; c < runtime->GetOutputCount(); ++c) {
    auto t = runtime->GetOutputTensor(c);

    TensorShape dst_desc = t->GetShape();
    dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);
    auto bytes = dst_desc.GetBytesIncludingPadding();
    vector<char> buffer(bytes);

    auto status = t->ConvertToHost((void*)buffer.data(), dst_desc);
    // ......
}
```
