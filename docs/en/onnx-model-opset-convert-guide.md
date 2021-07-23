Most `PPLNN` supported ops are based on onnx opset 11. If you are using onnx model with different opset version, you need to convert your onnx model opset version to 11.

## Use ONNX Official Opset Convert Tool

`ONNX` officially provided an opset convert tool `version_converter`. Its tutorials is at: [Version Conversion](https://github.com/onnx/tutorials/blob/master/tutorials/VersionConversion.md)

`version_converter` is easy to use:

```Python
import onnx
from onnx import version_converter

model = onnx.load("<your_path_to_onnx_model>")
converted_model = version_converter.convert_version(onnx_model, 11)
onnx.save(converted_model, "<your_save_path>")
```

### Error about Initializers Undefined

`ONNX version_converter` may have trouble when one of a node's input is initializer and the initializer is not a graph input(sometimes we may remove initializer from graph input for graph optimization). It is very likely to happen when a model has initializers. You may see error log like this:

```
IndexError: Input XXX is undefined!
```

You can try to use our script [convert_onnx_opset_version.py](../../tools/convert_onnx_opset_version.py) to solve this problem. Here's an example of `convert_onnx_opset_version.py`:

```Bash
python convert_onnx_opset_version.py --input_model input_model.onnx --output_model output_model.onnx --output_opset 11
```

this script is a wrapper of `ONNX version_converter`, first move all intializers to constant op to avoid the problem above, then run `ONNX version_converter`, finally change constant op back to initializers.

### Error about Initializers with Same Name

For `ONNX` version <= 1.8.0, there's another bug when attributes convert to initializers across different opset. For example, when convert op Clip from opset 10 to opset 11, attribute 'min' and 'max' will move to initializer with name 'min' and 'max'. If there're two or more Clip ops, there will be duplicated initializers with same name 'min' and 'max', and may cause errors.

`ONNX` has fixed this bug when version >= 1.8.1. If you meet this trouble, please update `ONNX` version to solve it.
