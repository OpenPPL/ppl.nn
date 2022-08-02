## How to Add a New Engine

There is an engine demo in [samples/cpp/engine](../../samples/cpp/engine).

##### 1. Define and Implement a Class Inherited from EngineImpl

`EngineImpl`(defined in [src/ppl/nn/engines/engine_impl.h](../../src/ppl/nn/engines/engine_impl.h)) defines the interfaces needed by `PPLNN`.

```c++
EngineContext* EngineImpl::CreateEngineContext();
```

Create an `EngineContext` used by a `Runtime` instance.

```c++
bool EngineImpl::Supports(const ir::Node* node) const;
```

Tell if this engine can run an op specified by `node`.

```c++
ppl::common::RetCode EngineImpl::ProcessGraph(const utils::SharedResource&, ir::Graph* graph,
                                              RuntimePartitionInfo* info);
```

Optimize `graph` and fill results into `info`.

`RuntimePartitionInfo` is defined as following:

```c++
struct RuntimePartitionInfo final {
    std::map<edgeid_t, RuntimeConstantInfo> constants;
    std::map<nodeid_t, std::unique_ptr<OptKernel>> kernels;
};
```

`constants` are read-only and used by multiple `Runtime` instances. `kernels` are a list of `OptKernel` that are used to create `KernelImpl` instances.

```c++
EngineImpl* Create() const;
```

Creates an instance of the same type as this engine.

##### 2. Define and Implement a Class Inherited from EngineContext

`EngineContext`(defined in [src/ppl/nn/engines/engine_context.h](../../src/ppl/nn/engines/engine_context.h)). An `EngineContext` is used by a `Runtime` instance only.

```c++
const char* EngineContext::GetName() const;
```

Returns the name of this `EngineContext`.

```c++
Device* EngineContext::GetDevice();
```

Returns the device instance used by a `Runtime`.

##### 3. Define and Implement Op Classes Inherited from OptKernel

`OptKernel`(defined in [src/ppl/nn/runtime/opt_kernel.h](../../src/ppl/nn/runtime/opt_kernel.h)) stores all data needed for evaluating an OP. It can create multiple `KernelImpl` instances.

```c++
KernelImpl* OptKernel::CreateKernelImpl() const;
```

Create a `KernelImpl` instance used in runtime stage.

##### 4. Define and Implement Op Classes Inherited from KernelImpl

`KernelImpl`(defined in [src/ppl/nn/runtime/kernel_impl.h](../../src/ppl/nn/runtime/kernel_impl.h)) is the main class used to evaluate an op, which is created by `OptKernel`. Each `KernelImpl` instance is used by only one `Runtime` instance.

```c++
ppl::common::RetCode KernelImpl::Execute(KernelExecContext* ctx);
```

## How to Add a new Op to an Existing Engine

We use the built-in `X86Engine` as an example to show how to add a new ONNX op.

##### Add a Parameter Parser

`ParamParserManager`(defined in [src/ppl/nn/models/onnx/param_parser_manager.h](../../src/ppl/nn/models/onnx/param_parser_manager.h)) is an derived class of `ppl::nn::utils::OpInfoManager`:

```c++
struct OpInfoManager {
public:
    ppl::common::RetCode Register(const std::string& domain, const std::string& type, const VersionRange& ver, const T& item);
    ...
};
```

which can be used to register parser routines for new ops:

```c++
typedef std::shared_ptr<ir::Attr> (*CreateParamFunc)();
typedef ppl::common::RetCode (*ParseParamFunc)(const ::onnx::NodeProto&, const ParamParserExtraArgs&, ir::Node*,
                                               ir::Attr*);
typedef ppl::common::RetCode (*PackParamFunc)(const ir::Node*, const ir::Attr*, ::onnx::NodeProto*);

struct ParserInfo final {
    CreateParamFunc create_param;
    ParseParamFunc parse_param;
    PackParamFunc pack_param;
};
```

If an op doesn't have any param, `parse_param` and `pack_param` of `ParserInfo` should be `nullptr`, or you can do some conversions in `parse_param`, e.g. mapping this type of op to another op type.

##### 1. Add a New Class Inherited from OptKernel

This depends on strategies the engine provides. In `X86Engine`, a Singleton `OptKernelCreatorManager` is used to manage `OptKernel` creator functions.

```c++
typedef X86OptKernel* (*OptKernelCreator)(const ir::Node*);

ppl::common::RetCode OptKernelCreatorManager::Register(
        const std::string& domain, const std::string& type, const utils::VersionRange versions,
        OptKernelCreator);
```

is used to register a function which is used to create an instance of the `OptKernel` instance.

##### 2. Add a New Class Inherited from KernelImpl

This is the same as adding a new class described in [How to Add a New Engine](#how-to-add-a-new-engine).
