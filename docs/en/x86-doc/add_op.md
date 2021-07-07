## Add Custom operator

This tutorial describes the steps needed to add a user-defined operator to PPL.NN on the x86 backend.

### 0. Overview

Here are the steps to add a user-defined operator on the x86 backend:

1) Operator parameter definition and parsing (If the operator does not require parameters, or the parameter definition and parsing functions are already added, skip this step)
2) Operator context implementation in the framework, including data type inference, shape inference, data format selection, etc.
3) Kernel execution scheduling
4) Kernel computing implementation

Note:

* \<opname\> is referred to as the name of the user-defined operator.
* The LeakyReLU operator has been choosen in this tutorial to demonstrate the implementation of a user-defined operator in PPL.NN. ***Some code will be omitted as needed during the introduction***. Please refer to the source code directly for the complete code if needed.
* The definition of operator LeakyReLU is available at ONNX documentation[https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu)

### 1. Operator Parameter Definition and Parsing

If the operator parameter definition and its parsing function have been added, skip this step.

#### 1.1. Add Operator Parameter Definition Struct

If the operator does not require parameters, skip this step.

Create file \<domain_name\>/\<opname\>_param.h in the ppl.nn/src/ppl/nn/params directory to define the parameter struct.
The `==` operator needs to be overloaded to support graph optimization.

Take LeakyReLU as an example, its parameter is defined in ppl.nn/src/ppl/nn/params/onnx/leaky_relu_param.h:

```c++
struct LeakyReLUParam {
    float alpha;  // LeakyReLU only requires one parameter "alpha"
    bool operator==(const LeakyReLUParam& p) const { return this->alpha == p.alpha; }   // overload operator ==
};
```

struct `LeakyReLUParam` defines the required parameters of the LeakyReLU operator. This operator only requires a `float` parameter `alpha`. In addition, `LeakyReLUParam` overloads the operator ==, which is used to determine whether two parameter objects are equal.

#### 1.2. Add Parameter Parsing Function

If the operator does not require parameters, skip this step.

Create file parse_\<opname\>\_param.h and parse_\<opname\>\_param.cc in the ppl.nn/src/ppl/nn/models/onnx/Parses directory, and place the parameter parsing function in it.

Take LeakyReLU as an example, its parameter parsing function is implemented in ppl.nn/src/ppl/nn/models/onnx/Parses/parse_leaky_relu_param.cc:

```c++
ppl::common::RetCode ParseLeakyReLUParam(const ::onnx::NodeProto& pb_node, void* arg, ir::Node*, ir::GraphTopo*) {
    auto param = static_cast<ppl::common::LeakyReLUParam*>(arg);
    param->alpha = utils::GetNodeAttrByKey<float>(pb_node, "alpha", 0.01);  // Parse the field "alpha" from onnx. If it does not exist, use the default value 0.01
    return ppl::common::RC_SUCCESS;
}
```

The function `ParseLeakyReLUParam` reads the onnx node information `pb_node`, parses the field "alpha" through `GetNodeAttrByKey`. Then puts it into the parameter structure defined above to complete the parameter parsing.

#### 1.3. Register Parameters and Parsing Function

The parameter structure and parsing function need to be registered into the `ParamParseManager()` function in ppl.nn/src/ppl/nn/models/onnx/param_Parse_manager.cc.

There are two macros can be used:

* `PPL_REGISTER_OP_WITHOUT_PARAM`: Used to register operators without parameters
* `PPL_REGISTER_OP_WITH_PARAM`: Used to register operators with parameters

Take LeakyReLU as an example:

```c++
PPL_REGISTER_OP_WITH_PARAM("", "LeakyRelu", ppl::nn::common::LeakyReLUParam, ParseLeakyReLUParam);
```

The first parameter is domain name, which must be same as domain in the onnx model; the second parameter "LeakyRelu" is op_type, which must be consistent with the op_type in the onnx model; the third and fourth parameters are parameter structure and parsing function respectively.

### 2. Operator Context Implementation in the Framework

Add files \<opname\>\_op.h and \<opname\>\_op.cc in the ppl.nn/src/ppl/nn/engines/x86/optimizer/\<domain_name\> directory to define and implement operator context class.

For example, the operator context class of LeakyReLU is placed in ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.h:

```c++
class LeakyReluOp final : public X86OptKernel {
public:
    LeakyReluOp(const ir::Node* node)
        : X86OptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;    // Initialize(required)
    KernelImpl* CreateKernelImpl() const override;                          // Create kernel interface(required)
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,          // format selection(optional)
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;

private:
    std::shared_ptr<ppl::common::LeakyReLUParam> param_;    // pointer to parameter struct(if the operator does not need parameters, this variable is not needed)
};
```

* `Init`(required): Perform initialization operations, such as loading parameters, registering data type inference functions, registering shape inference functions, etc.
* `CreateKernelImpl`(required): Create kernel interface object
* `SelectFormat`(optional): do data format(or called layout) selection
* `param_`(optional): if the operator does not need parameters, this variable is not needed

Some member functions and variables can be changed according to actual needs.

#### 2.1. Register Data Type Inference Function

The data type inference function is used to infer the output data types based on the input data types.

The data type inference function needs to be registered to `infer_type_func_` in the `Init` function.
`infer_type_func_` is a std::function object, input InputOutputInfo*, and return void.
You can use function or lambda expressions to define data type inference functions, and then assign them to `infer_type_func_` to complete the registration.

For example, the data type inference function of LeakyReLU is registerred in the `Init` function of ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
infer_type_func_ = GenericInferType;
```

`GenericInferType` is a data type inference function predefined by the framework. It is defined in ppl.nn/src/ppl/nn/engines/x86/optimizer/opt_kernel.h:

```c++
static void GenericInferType(InputOutputInfo* info) {   // All output data types are consistent with the first input data type
    auto& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
        auto out_shape = &info->GetOutput<TensorImpl>(i)->GetShape();
        out_shape->SetDataType(in_shape0.GetDataType());
    }
}
```

You can register pre-defined data type inference functions or register functions written by yourself according to your op's need.

#### 2.2. Register Shape Inference Function

The shape inference function is used to infer the output shapes according to the input shapes.

The shape inference function needs to be registered to `infer_dims_func_` in the `Init` function.
`infer_dims_func_` is a std::function object, input InputOutputInfo*, and return ppl::common::RetCode.
You can use function or lambda expressions to define shape inference functions, and then assign them to `infer_dims_func_` to complete the registration.

For example, the shape inference function of LeakyReLU is registers in the `Init` function of ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
    return oputils::ReshapeLeakyReLU(info, nullptr);
};
```

`ReshapeLeakyReLU`function is implemented in ppl.nn/src/ppl/nn/oputils/onnx/reshape_leaky_relu.cc:

```c++
RetCode ReshapeLeakyReLU(InputOutputInfo* info, const void*) {
    if (info->GetInputCount() != 1 || info->GetOutputCount() != 1) {
        return RC_INVALID_VALUE;
    }
    const TensorShape& in_shape0 = info->GetInput<TensorImpl>(0)->GetShape();
    auto out_shape0 = &info->GetOutput<TensorImpl>(0)->GetShape();
    if (in_shape0.IsScalar()) {
        out_shape0->ReshapeAsScalar();
    } else {
        out_shape0->Reshape(in_shape0.GetDims(), in_shape0.GetDimCount());
    }
    return RC_SUCCESS;
}
```

You can either write a separate reshape function under `ppl.nn/src/ppl/nn/oputils` like LeakyReLU, or you can write all the code directly in the Init function.

#### 2.3. Add Data Format Selection Function

Data format selection function `SelectFormat` selects the operator's **needed** input data format and output data format according to the operator's parameters, input data type, data format, shape, kernel support and other information.

The input data format of the operator and the ***needed*** input data format can be different.
The input data format is the actual input data format before format selection of this operator (usually determined by the output of the previous operator or the input of the network);
The ***needed*** input data format is the data format selected according to some informations such as parameters, input data type, data format, shape, and kernel support.
When these two are different, the framework will automatically insert a reorder operator to convert different data format.

now x86 architecture supports:
* NDARRAY (For 4 dimensions, it is NCHW)
* N16CX (For 4 dimensions, it is N16CHW, or called N16CHWc16, NCHWc16, etc.)

Take LeakyReLU as an example, its data format selection function is in ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
RetCode LeakyReluOp::SelectFormat(const InputOutputInfo& info,
                                  vector<dataformat_t>* selected_input_formats, // needed input data format, default is all NDARRAY
                                  vector<dataformat_t>* selected_output_formats // output data format, default is all NDARRAY
    ) {
    if (info.GetInput<TensorImpl>(0)->GetShape().GetDataFormat() == DATAFORMAT_N16CX) { // operator's input data format
        selected_input_formats->at(0) = DATAFORMAT_N16CX;
        selected_output_formats->at(0) = DATAFORMAT_N16CX;
    }
    return RC_SUCCESS;
}
```

`selected_input_formats` is the needed input data format, selected_output_formats is the output data format. All default values are NDARRAY.
Since LeakyReLU implements NDARRAY & N16CX version kernel, when the input data format is N16CX, the `selected_input_formats` and `selected_output_formats` use N16CX; when the input is in the NDARRAY format, selected formats use NDARRAY, and the function does not do any action.

If your custom operator only implements the NDARRAY version (input and output only support NDARRAY format), the `SelectFormat` function can be omitted, and the default function of the base class will be used. NDARRAY will be selected.

#### 2.4. Add CreateKernelImpl

The `CreateKernelImpl` function is used to create kernel interface. There are two functions can be used:

* `CreateKernelImplWithoutParam`: Used for operators without parameters
* `CreateKernelImplWithParam`: Used for operators with parameters. The pointer of the parameter structure is needed

LeakyReLU uses `CreateKernelImplWithParam` located in ppl.nn/src/ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
KernelImpl* LeakyReluOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LeakyReluKernel>(param_.get());    // The pointer of the parameter is needed
}
```

#### 2.5. Register Operator Context Class

The operator context class is registerred using the macro `REGISTER_OPT_KERNEL_CREATOR` to `OptKernelCreatorManager()` located in ppl.nn/src/ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.cc.

Take LeakyReLU as an example:

```c++
REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", LeakyReluOp);
```

The first parameter is domain name; the second parameter is op_type; the third parameter is the name of the operator context class defined above.

### 3. Kernel Execution Scheduling

Create \<opname\>\_kernel.h and \<opname\>\_kernel.cc in ppl.nn/src/ppl/nn/engines/x86/kernels/\<domain_name\> directory to define and implement the operator kernel scheduling interface.

Take LeakyReLU as an example, the kernel interface is defined in ppl.nn/src/ppl/nn/engines/x86/kernels/onnx/leaky_relu_kernel.h:

```c++
class LeakyReluKernel : public X86Kernel {
public:
    LeakyReluKernel(const ir::Node* node) : X86Kernel(node) {}
    void SetParam(const ppl::nn::common::LeakyReLUParam* p) { param_ = p; } // not needed when there's no parameter
private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;    // kernel execute function
private:
    const ppl::nn::common::LeakyReLUParam* param_ = nullptr;    // not needed when there's no parameter
};
```

Class member  `SetParam` and `param_` can be ignored if the operator parameter is not required.

#### 3.1. Add DoExecute Function

The `DoExecute` function reads the operator inputs, executes the kernel function, and produces operator outputs.

Take LeakyReLU as an example, its `DoExecute` function locates in ppl.nn/src/ppl/nn/engines/x86/kernels/onnx/leaky_relu_kernel.cc:

```c++
ppl::common::RetCode LeakyReluKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);  // get input tensor
    auto y = ctx->GetOutput<TensorImpl>(0); // get output tensor

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());   // print debug info, only valid in Debug mode.
    PPLNN_X86_DEBUG_TRACE("Input [x]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_X86_DEBUG_TRACE("Output [y]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_X86_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape().GetDataType();

    if (data_type == ppl::common::DATATYPE_FLOAT32) {   // Data type judgment
        if (MayUseISA(ppl::common::ISA_X86_AVX)) {  // whether support avx instruction set
            return kernel::x86::leaky_relu_fp32_avx(&y->GetShape(), x->GetBufferPtr<float>(),   // execute avx kernel function
                                                    param_->alpha, y->GetBufferPtr<float>());
        } else if (MayUseISA(ppl::common::ISA_X86_SSE)) {   // whether support sse instruction set
            return kernel::x86::leaky_relu_fp32_sse(&y->GetShape(), x->GetBufferPtr<float>(),   // execute sse kernel function
                                                    param_->alpha, y->GetBufferPtr<float>());
        } else {
            LOG(ERROR) << "get unsupported isa " << GetISA();
        }
    } else {
        LOG(ERROR) << "unsupported data type: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}
```

Macro `PPLNN_X86_DEBUG_TRACE` and `PPL_X86_TENSOR_PRINT_DEBUG_MSG` are used to print debug informations, which is only valid in Debug mode.

The `MayUseISA` function is used to determine whether the current environment can execute the specified ISA, so as to call different kernel functions.
Commonly used ISAs are: AVX512, FMA, AVX, SSE. Thus operators implemented with scalar code do not need this function.

#### 3.2. Add CanDoExecute Function

The `CanDoExecute` function is executed before `DoExecute` to detect whether `DoExecute` can be executed.

Most operators use the implementation of the base class defined in ppl.nn/src/ppl/nn/engines/x86/kernel.cc:

```c++
bool X86Kernel::CanDoExecute(const KernelExecContext& ctx) const {  // If there is an empty tensor in the input, return false, otherwise return true
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape().GetBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}
```

In most cases, this function does not need to be overriden. Overide it when kernel execution condition needs specified.

### 4. Kernel Computing Implementation

Operator kernel computing is implemented and fully optimized in the x86 kernel function located in ppl.nn/src/ppl/nn/engines/x86/impls directory.

As the kernel computing implementations are highly decoupled with the upper framework, the code structure of kernel implementations can be freely designed according to the characteristics of the user-defined operator. In this tutorial only a general reference for writing kernel functions is provided.

#### 4.1. Kernel Function Declaration

We recommend to place the declarations of kernel functions in ppl.nn/src/ppl/nn/engines/x86/impls/include/ppl/kernel/x86 directory. The recommended file path is ppl.nn/src/ppl/nn/engines/x86/impls/include/ppl/kernel/x86/\<data_type\>/\<opname\>.h

The input parameters of functions can be defined freely as needed. It's recommended to return a ppl::common::RetCode to indicate the function execution status.

Function naming convention: \<opname\>\_\<data_format\>\_\<specialization_desctription\>\_\<data_type\>\_\<isa_type\>

For example, the kernel function declaration of `LeakyReLU` is placed in ppl.nn/src/ppl/nn/engines/x86/impls/include/ppl/kernel/x86/fp32/leaky_relu.h with the name of  `leaky_relu_fp32_avx` and `leaky_relu_fp32_sse`:

```c++
ppl::common::RetCode leaky_relu_fp32_avx(   // avx kernel function declaration
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);

ppl::common::RetCode leaky_relu_fp32_sse(   // sse kernel function declaration
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);
```

Since LeakyReLU's implementation supports any data format, we do not use \<data_format\> to name kernel function.

#### 4.2. Kernel Function Implementation

The implementation of the kernel function is suggested to be placed in the ppl.nn/src/ppl/nn/engines/x86/impls/src/ppl/kernel/x86 directory. It is recommended to create a separate directory for each operator.
The different ISA code should be placed in different .cpp files and distinguished by the ISA architecture name when naming the files.

The recommended file path is ppl.nn/src/ppl/nn/engines/x86/impls/src/ppl/kernel/x86/\<data_type\>/\<opname\>/\<opname\>\_\< data_type\>\_<isa_type>.cpp. This file naming convention can be recognized by ppl.nn/src/ppl/nn/engines/x86/impls/CMakeLists.txt and automatically add the corresponding ISA compilation definitions.

Take LeakyReLU as an example. This operator implements the avx and sse version of fp32, so two files, leaky_relu_fp32_avx.cpp and leaky_relu_fp32_avx.cpp, are created in the ppl.nn/src/ppl/nn/engines/x86/impls/src/ppl/kernel/x86/fp32/leaky_relu directory, in which kernel functions `leaky_relu_fp32_avx` and `leaky_relu_fp32_sse` are implemented respectively.

### 5. Tips

#### 5.1. Compilation after Implementing User-defined Operators

Since there are several newly added .cpp files, it's required to delete CMakeCache.txt and re-run cmake, otherwise linking error will occurr.

If the file name of kernel function cannot be recognized by ppl.nn/src/ppl/nn/engines/x86/impls/CMakeLists.txt, an error of ISA not support may occur.

#### 5.2. Other Reference Examples

Other user-defined operators may be different from `LeakyReLU` in this tutorial, we provide more operator implementations for reference. The code structure of these operators is similar to that of `LeakyReLU`. Choose appropriate references accordingly:

| ref ops           | has parameters | support NDARRAY | support N16CX | support multi-ISA | run when there's an empty input | process weight offline |
|-------------------|----------------|-----------------|---------------|-------------------|---------------------------------|------------------------|
| Exp/Tanh          |                | &check;         | &check;       | &check;           |                                 |                        |
| LeakyReLU/Softmax | &check;        | &check;         | &check;       | &check;           |                                 |                        |
| Tile/TopK         | &check;        | &check;         |               |                   |                                 |                        |
| Clip/Resize       | &check;        | &check;         | &check;       | &check;           | &check;                         |                        |
| Conv/FC           | &check;        | &check;         | &check;       | &check;           |                                 | &check;                |
