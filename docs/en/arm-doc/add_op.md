## Add Custom operator

This tutorial describes the steps needed to add a user-defined operator to PPL.NN on the ARM backend.

### 0. Overview

Here are the steps to add a user-defined operator on the ARM backend:

1) Operator parameter definition and parsing (If the operator does not require parameters, or the parameter definition and parsing functions are already added, skip this step)
2) Operator context implementation in the framework, including data type inference, shape inference, data type selection, data format selection, etc.
3) Kernel execution scheduling
4) Kernel computing implementation

Note:

* \<opname\> is referred to as the name of the user-defined operator.
* The LeakyReLU operator has been choosen in this tutorial to demonstrate the implementation of a user-defined operator in PPL.NN. ***Some code will be omitted as needed during the introduction***. Please refer to the source code directly for the complete code if needed.
* The definition of operator LeakyReLU is available at ONNX documentation[https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyReLU](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyReLU)

### 1. Operator Parameter Definition and Parsing

To add operator parameter & parsing function, please refer to the corresponded chapter of [add custom operator for x86](../x86-doc/add_op.md).

If the operator parameter definition and its parsing function have been added, skip this step.

### 2. Operator Context Implementation in the Framework

Add files \<opname\>\_op.h and \<opname\>\_op.cc in the ppl.nn/src/ppl/nn/engines/arm/optimizer/\<domain_name\> directory to define and implement operator context class.

For example, the operator context class of LeakyReLU is placed in ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.h:

```c++
class LeakyReLUOp final : public ArmOptKernel {
public:
    LeakyReLUOp(const ir::Node* node)
        : ArmOptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;    // Initialize(required)
    KernelImpl* CreateKernelImpl() const override;                          // Create kernel interface(required)
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,        // data type selection(optional)
                                        std::vector<ppl::common::datatype_t>* selected_input_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_types,
                                        const ppl::common::datatype_t preferred_fp_datatype) override;
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
* `SelectDataType`(optional): do data type selection
* `param_`(optional): if the operator does not need parameters, this variable is not needed

Some member functions and variables can be changed according to actual needs.

#### 2.1. Register Data Type Inference Function

The data type inference function is used to infer the output data types based on the input data types.

The data type inference function needs to be registered to `infer_type_func_` in the `Init` function.
`infer_type_func_` is a std::function object, input InputOutputInfo*, and return void.
You can use function or lambda expressions to define data type inference functions, and then assign them to `infer_type_func_` to complete the registration.

For example, the data type inference function of LeakyReLU is registerred in the `Init` function of ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
infer_type_func_ = GenericInferType;
```

`GenericInferType` is a data type inference function predefined by the framework. It is defined in ppl.nn/src/ppl/nn/engines/arm/optimizer/opt_kernel.h:

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

For example, the shape inference function of LeakyReLU is registers in the `Init` function of ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc:

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

The input data format of the operator and the **needed** input data format can be different.
The input data format is the actual input data format before format selection of this operator (usually determined by the output of the previous operator or the input of the network);
The **needed** input data format is the data format selected according to some informations such as parameters, input data type, data format, shape, and kernel support.
When these two are different, the framework will automatically insert a reorder operator to convert different data format.

now ARM architecture supports:
* NDARRAY (For 4 dimensions, it is NCHW)
* N4CX (For 4 dimensions, it is N4CHW, or called N4CHWc4, NCHWc4, etc.): only used by fp32
* N8CX (For 4 dimensions, it is N8CHW, or called N8CHWc8, NCHWc8, etc.): only used by fp16

Take LeakyReLU as an example, its data format selection function is in ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
RetCode LeakyReLUOp::SelectFormat(const InputOutputInfo& info,
                                  std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                  std::vector<ppl::common::dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = selected_output_formats->at(0) = // select the same data format as operator's input data format
        info.GetInput<TensorImpl>(0)->GetShape()->GetDataFormat();
    return RC_SUCCESS;
}
```

`selected_input_formats` is the needed input data format, `selected_output_formats` is the output data format.

If your custom operator only implements the NDARRAY version (input and output only support NDARRAY format), the `SelectFormat` function can be omitted, and the default function of the base class will be used. NDARRAY will be selected.

#### 2.4. Add Data Type Selection Function

Data type selection function `SelectData` selects the operator's **needed** input data type and output data type according to the operator's parameters, input data type, data format, shape, kernel support and other information. It is similar to `SelectFormat`.
When operator's input data type are different with operator's **needed** input data type, the framework will automatically insert a reorder operator to convert different data type.

Take LeakyReLU as an example, its data type selection function is in ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
RetCode LeakyReLUOp::SelectDataType(const InputOutputInfo& info,
                                    std::vector<ppl::common::datatype_t>* selected_input_types,
                                    std::vector<ppl::common::datatype_t>* selected_output_types,
                                    const ppl::common::datatype_t preferred_fp_datatype) { // preferred floating point data type, only fp16 & fp32 are supported now
    GenericSelectDataType(info, selected_input_types, selected_output_types, preferred_fp_datatype); // generic data type selection function, defined in ppl.nn/src/ppl/nn/engines/arm/optimizer/opt_kernel.h
    return RC_SUCCESS;
}
```

#### 2.5. Add CreateKernelImpl

The `CreateKernelImpl` function is used to create kernel interface. There are two functions can be used:

* `CreateKernelImplWithoutParam`: Used for operators without parameters
* `CreateKernelImplWithParam`: Used for operators with parameters. The pointer of the parameter structure is needed

LeakyReLU uses `CreateKernelImplWithParam` located in ppl.nn/src/ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.cc:

```c++
KernelImpl* LeakyReLUOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<LeakyReLUKernel>(param_.get());    // The pointer of the parameter is needed
}
```

#### 2.6. Register Operator Context Class

The operator context class is registerred using the macro `REGISTER_OPT_KERNEL_CREATOR` to `OptKernelCreatorManager()` located in ppl.nn/src/ppl/nn/engines/arm/optimizer/opt_kernel_creator_manager.cc.

Take LeakyReLU as an example:

```c++
REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", 6, 16, LeakyReLUOp);
```

The first parameter is domain name; the second parameter is op_type; the third/forth parameter are supported min/max opset version respectively; the last parameter is the name of the operator context class defined above.

### 3. Kernel Execution Scheduling

Create \<opname\>\_kernel.h and \<opname\>\_kernel.cc in ppl.nn/src/ppl/nn/engines/arm/kernels/\<domain_name\> directory to define and implement the operator kernel scheduling interface.

Take LeakyReLU as an example, the kernel interface is defined in ppl.nn/src/ppl/nn/engines/arm/kernels/onnx/leaky_relu_kernel.h:

```c++
class LeakyReLUKernel : public ArmKernel {
public:
    LeakyReLUKernel(const ir::Node* node) : ArmKernel(node) {}
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

Take LeakyReLU as an example, its `DoExecute` function locates in ppl.nn/src/ppl/nn/engines/arm/kernels/onnx/leaky_relu_kernel.cc:

```c++
ppl::common::RetCode LeakyReLUKernel::DoExecute(KernelExecContext* ctx) {
    auto x = ctx->GetInput<TensorImpl>(0);  // get input tensor
    auto y = ctx->GetOutput<TensorImpl>(0); // get output tensor

    PPLNN_ARM_DEBUG_TRACE("Op: %s\n", GetName().c_str());   // print debug info, only valid in Debug mode.
    PPLNN_ARM_DEBUG_TRACE("Input [x]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(x);
    PPLNN_ARM_DEBUG_TRACE("Output [y]:\n");
    PPL_ARM_TENSOR_PRINT_DEBUG_MSG(y);
    PPLNN_ARM_DEBUG_TRACE("alpha: %f\n", param_->alpha);
    PPLNN_ARM_DEBUG_TRACE("isa: %u\n", GetISA());

    const auto data_type = x->GetShape()->GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {   // fp32 implementation
        if (MayUseISA(ppl::common::ISA_ARMV8)) {
            return ppl::kernel::arm_server::neon::leaky_relu_fp32(x->GetShape(), x->GetBufferPtr<float>(),
                                                                  param_->alpha, y->GetBufferPtr<float>());
        }
    } 
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (data_type == ppl::common::DATATYPE_FLOAT16) {  // fp16 implementation, need armv8.2-a ISA support
        if (MayUseISA(ppl::common::ISA_ARMV8_2)) {
            return ppl::kernel::arm_server::neon::leaky_relu_fp16(x->GetShape(), x->GetBufferPtr<__fp16>(),
                                                                  param_->alpha, y->GetBufferPtr<__fp16>());
        }
    } 
#endif
    else {
        LOG(ERROR) << "unsupported datatype: " << ppl::common::GetDataTypeStr(data_type) << ".";
    }

    return ppl::common::RC_UNSUPPORTED;
}
```

Macro `PPLNN_ARM_DEBUG_TRACE` and `PPL_ARM_TENSOR_PRINT_DEBUG_MSG` are used to print debug informations, which is only valid in Debug mode.

The `MayUseISA` function is used to determine whether the current environment can execute the specified ISA, so as to call different kernel functions.
Now PPL.NN ARM support: ARMV8(default support), ARMV8_2(need compiler to support armv8.2-a ISA). 

#### 3.2. Add CanDoExecute Function

The `CanDoExecute` function is executed before `DoExecute` to detect whether `DoExecute` can be executed.

Most operators use the implementation of the base class defined in ppl.nn/src/ppl/nn/engines/arm/kernel.cc:

```c++
bool ArmKernel::CanDoExecute(const KernelExecContext& ctx) const {  // If there is an empty tensor in the input, return false, otherwise return true
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape().CalcBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}
```

In most cases, this function does not need to be overriden. Overide it when kernel execution condition needs specified.

### 4. Kernel Computing Implementation

Operator kernel computing is implemented and fully optimized in the ARM kernel function located in ppl.nn/src/ppl/nn/engines/arm/impls directory.

As the kernel computing implementations are highly decoupled with the upper framework, the code structure of kernel implementations can be freely designed according to the characteristics of the user-defined operator. In this tutorial only a general reference for writing kernel functions is provided.

#### 4.1. Kernel Function Declaration

We recommend to place the declarations of kernel functions in ppl.nn/src/ppl/nn/engines/arm/impls/include/ppl/kernel/arm_server directory. The recommended file path is ppl.nn/src/ppl/nn/engines/arm/impls/include/ppl/kernel/arm_server/\<opname\>/\<isa\>/\<opname\>.h

The input parameters of functions can be defined freely as needed. It's recommended to return a `ppl::common::RetCode` to indicate the function execution status.

Function naming convention: \<opname\>\_\<data_format\>\_\<specialization_desctription\>\_\<data_type\>

For example, the kernel function declaration of `LeakyReLU` is placed in ppl.nn/src/ppl/nn/engines/arm/impls/include/ppl/kernel/arm_server/leaky_relu/neon/leaky_relu.h with the name of  `leaky_relu_fp32` and `leaky_relu_fp16`:

```c++
ppl::common::RetCode leaky_relu_fp32(
    const ppl::nn::TensorShape *src_shape,
    const float *src,
    const float alpha,
    float *dst);

#ifdef PPLNN_USE_ARMV8_2_FP16  // need compiler to support armv8.2-a ISA
ppl::common::RetCode leaky_relu_fp16(
    const ppl::nn::TensorShape *src_shape,
    const __fp16 *src,
    const float alpha,
    __fp16 *dst);
#endif
```

Since LeakyReLU's implementation supports any data format, we do not use \<data_format\> to name kernel function.

#### 4.2. Kernel Function Implementation

The implementation of the kernel function is suggested to be placed in the ppl.nn/src/ppl/nn/engines/arm/impls/src/ppl/kernel/arm_server directory. It is recommended to create a separate directory for each operator.

The recommended file path is ppl.nn/src/ppl/nn/engines/arm/impls/src/ppl/kernel/arm_server/\<opname\>/\<isa\>/\<opname\>.cpp.

Take LeakyReLU as an example. This operator implements the neon version, so leaky_relu.cpp is created in the ppl.nn/src/ppl/nn/engines/arm/impls/src/ppl/kernel/arm_server/leaky_relu/neon directory, in which kernel functions `leaky_relu_fp32` and `leaky_relu_fp16` are implemented.

### 5. Tips

#### 5.1. Compilation after Implementing User-defined Operators

Since there are several newly added .cpp files, it's required to delete CMakeCache.txt and re-run cmake, otherwise linking error will occurr.

#### 5.2. FP16 Compile Definitions

All code used armv8.2-a instruction(typically FP16 instruction) should be placed in compile definition `PPLNN_USE_ARMV8_2_FP16`, in order to pass compile when the compiler does not support armv8.2-a ISA.

#### 5.3. Other Reference Examples

Other user-defined operators may be different from `LeakyReLU` in this tutorial, we provide more operator implementations for reference. The code structure of these operators is similar to that of `LeakyReLU`. Choose appropriate references accordingly:

| ref ops           | has parameters | support NDARRAY | support N4CX/N8CX | run when there's an empty input | process weight offline |
|-------------------|----------------|-----------------|-------------------|---------------------------------|------------------------|
| Exp/Sigmoid       |                | &check;         | &check;           |                                 |                        |
| LeakyReLU/Softmax | &check;        | &check;         | &check;           |                                 |                        |
| Tile/TopK         | &check;        | &check;         |                   |                                 |                        |
| Clip/Resize       | &check;        | &check;         | &check;           | &check;                         |                        |
| Conv/FC           | &check;        | &check;         | &check;           |                                 | &check;                |
