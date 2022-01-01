## Add custom operator
Adding a custom operator requires adding the operator definition and its implementation on different architectures.

### 0. Overview
The steps for PPLNN to add a custom operator to the riscv architecture are as follows:
1. Add operator parameter definition
1. Add operator definition, including data type inference, shape inference, data format, etc.
2. Add operator interface
3. Add kernel function

### 1. Add operator parameter definition
For the addition of operator parameters, please refer to the corresponding chapter of [add custom operator for x86](../x86-doc/add_op.md).

### 2. Add operator definition

Add <opname\>_op.h and <opname\>_op.cc in the ppl.nn/src/ppl/nn/engines/riscv/optimizer/ops/onnx directory to define and implement operators.

Take Clip as an example, its operator definition class is in ppl.nn/src/ppl/nn/engines/riscv/optimizer/ops/onnx/clip_op.h:
``` c++
class ClipOp final: public RiscvOptKernel {
public:
    ClipOp(const ir::Node* node): RiscvOptKernel(node) {}
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                        std::vector<ppl::common::datatype_t>* selected_input_data_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_data_types) override;
};
```

#### 2.1 Register Shape Inference Function

The shape inference function is used to infer the output data shapes according to the input data shapes.

The shape inference function needs to be registered in the `infer_dims_func_` in the `Init` function. `infer_dims_func_` is a std::function object, input InputOutputInfo*, and return ppl::common::RetCode. You can use functions or lambda expressions to define the shape inference function, and then assign it to `infer_dims_func_` to complete the registration.

Clip uses a default shape inference function `GenericInferDims`. The specific implementation of this function can be found in src/ppl/nn/engines/riscv/optimizer/opt_kernel.h.

#### 2.2 Write data type selection function

The data type selection function `SelectFormat` selects the input data type required by the operator and the output data type according to information such as parameters, input data type, format, shapes, and type supported by the operator's bottom layer.

The input type of the operator and the input type required by the operator can be different. The input type of the operator refers to the true type of the input data before the type selection of the operator (usually determined by the output of the previous operator or the input of the network). The input type required by the operator refers to the operator based on Parameters, input data type, format, shapes, type supported by the bottom layer of the operator and other information, select the input type required by the operator. When the two are different, the framework will automatically insert a type conversion operator to convert different data types.

Take Clip as an example, the specific usage method is as follows:
``` c++
RetCode ClipOp::SelectDataType(const InputOutputInfo& info,
                               std::vector<datatype_t>* selected_input_data_types,
                               std::vector<datatype_t>* selected_output_data_types) {

    if (DATATYPE_FLOAT16 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT16;
    } else if (DATATYPE_FLOAT32 == selected_input_data_types->at(0)) {
        selected_output_data_types->at(0) = DATATYPE_FLOAT32;
    }
    return RC_SUCCESS;
}
```

`selected_input_data_types` contains the input data types of the operator, and the elements of `selected_output_data_types` are FP32 by default. In the `SelectDataType` function, store the input data type required by the operator and the output data type required by the operator into `selected_input_data_types` and `selected_output_data_types` respectively.

Due to the need to support mixed precision, the data type inference function is not used in the riscv architecture (similar to the shape inference function, defined in the `Init` function and assigned to `infer_type_func_`), but still need to be the default in the `Init` function The data type inference function `GenericInferType` is assigned to `infer_type_func_`.

#### 2.3 Write data format selection function

The data format selection function SelectFormat selects the input data format required by the operator and the output data format according to information such as parameters, input data type, format, shapes, and format supported by the operator's bottom layer.

Similar to the data type of the operator, the input data format of the operator and the needed input data format can be different. The input data format is the actual input data format before format selection of this operator (usually determined by the output of the previous operator or the input of the network); The needed input data format is the data format selected according to some informations such as parameters, input data type, data format, shape, and kernel support. When these two are different, the framework will automatically insert a reorder operator to convert different data format.

The current data format supported by the riscv architecture: NDARRAY, N4CX, N8CX.

Take Clip as an example, the specific usage method is as follows:
``` c++
RetCode ClipOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                             vector<dataformat_t>* selected_output_formats) {
    if (DATAFORMAT_N8CX == selected_input_formats->at(0)) {
        selected_output_formats->at(0) = DATAFORMAT_N8CX;
    } else if(DATAFORMAT_N4CX == selected_input_formats->at(0)) {
        selected_output_formats->at(0) = DATAFORMAT_N4CX;
    }

    return RC_SUCCESS;
}
```
`selected_input_formats` contains the input data format of the operator, and elements of `selected_output_formats` are NDARRAY by default. In the `SelectFormat` function, the input data format required by the operator and the output data format required by the operator are written into `selected_input_formats` and `selected_output_formats` respectively.

#### 2.4 Add CreaeteKernelImpl

The `CreateKernelImpl` function is used to create a kernel interface. According to whether the operator requires parameters, two functions can be used:
`CreateKernelImplWithoutParam`: used for operators without parameters
`CreateKernelImplWithParam`: used for operators that require parameters, and need to pass in the pointer to the parameter structure

Clip is an operator without parameters, implemented as follows:
``` c++
KernelImpl* TestOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<TestKernel>();
}
```

#### 2.5 Registration operator definition

The operator context class is registerred using the macro `REGISTER_OPT_KERNEL_CREATOR` to `OptKernelCreatorManager()` located in ppl.nn/src/ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.cc.

The operator of Clip is registered as follows:
``` c++
REGISTER_OPT_KERNEL_CREATOR("", "Clip", 7, 11, ClipOp);
```

The first parameter is domain name; the second parameter is op_type; the third and fourth parameters indicate the range of opsets supported by the op. For example, in this example, Clip supports opset(7)~opset(11); the fourth parameter Define the name of the class for the operator defined above.

### 3. Add operator interface

Add <opname\>_kernel.h and <opname\>_kernel.cc to the ppl.nn/src/ppl/nn/engines/riscv/kernels/onnx directory to define and implement the operator call interface.

The operator call interface of Clip is defined as follows:
``` c++
class ClipKernel: public RiscvKernel {
public:
    ClipKernel(const ir::Node* node): RiscvKernel(node) {}

private:
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;
};
```

If the operator has no parameters, `SetParam` and `param_` do not need to be added, just the constructor and the `DoExecute` function. The `DoExecute` function can get the input and output information of the operator from the parameters, which is used to call a custom kernel function and write the result to the output tensor.

The `CanDoExecute` function is executed before `DoExecute` and is used to determine whether `DoExecute` can be executed. In most cases, this function does not need to be overriden. If you need to use a behavior different from the base class, you need to override this function.

### 4. Add kernel function

Operator kernel computing is implemented and fully optimized in the riscv kernel function located in ppl.nn/src/ppl/nn/engines/riscv/impls directory.

As the kernel computing implementations are highly decoupled with the upper framework, the code structure of kernel implementations can be freely designed according to the characteristics of the user-defined operator. In this tutorial only a general reference for writing kernel functions is provided.

#### 4.1 kernel function declaration

The interface declaration of the kernel function is unified in the ppl.nn/src/ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv directory, and placed in different subdirectories according to the data type. The recommended path is ppl.nn/src/ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv/<data_type\>/<opname\>.h.

#### 4.2 kernel function implementation

The implementation of the kernel function is placed in the ppl.nn/src/ppl/nn/engines/riscv/impls/src/ppl/kernel/riscv directory, and placed in different subdirectories according to the data type. Since the implementation of the kernel function may require multiple files, it is recommended to create a separate directory for each operator. The recommended file path is ppl.nn/src/ppl/nn/engines/riscv/impls/src/ppl/kernel/riscv/<data_type\>/<opname\>/<opname\>_<data_type\>.cpp.