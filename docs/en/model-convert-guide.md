`PPLNN` supports the ONNX format as the input, and there are plenty of ways to get ONNX models:
* Download pre-trained models directly from the official [ONNX Model Zoo](https://github.com/onnx/models).
* Convert models from various machine learning frameworks, i.e. PyTorch, TensorFlow, Caffe, etc.
* Convert models from other toolboxes, i.e. OpenMMLab.

This section takes OpenMMLab and PyTorch as examples, describing how to convert a model defined in the frameworks into an ONNX model. Converting from other frameworks can refer to the [offical ONNX tutorial](https://github.com/onnx/tutorials).


## Convert Model from OpenMMLab

[OpenMMLab](https://github.com/open-mmlab) is a series of open-source toolboxes and benchmarks based on PyTorch and covers a wide range of CV tasks. We strongly recommend training models using OpenMMLab.

Every member in OpenMMLab provides a converting tool in the project. After installing the corresponding OpenMMLab toolboxes, you can use this command to convert the pre-trained model to the ONNX format
```bash
python tools/deployment/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --output_file ${ONNX_FILE} [--shape ${INPUT_SHAPE} --dynamic-export --verify]
```
Here we take two models for different tasks as simple examples.

### Example: Converting MobileNetV2

If you have not installed MMClassification, following this [tutorial](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md) to install it first. MMClassification provides a pre-trained MobileNetV2 in the model zoo, we will download this checkpoint and convert it into an ONNX model.

```bash
cd mmclassification && mkdir checkpoints && cd checkpoints
wget https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth
```

Use the given converting tool to convert the checkpoint to an ONNX model.

```bash
python ../tools/deployment/pytorch2onnx.py ../configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py \
--checkpoint mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth \
--output-file mobilenet_v2.onnx --simplify
```

An ONNX model named `mobilenet_v2.onnx` will be generated in the current directory.
Here are some useful flags during conversion:
* --output-file &emsp; Specify the name of output file, the default name is 'tmp.onnx'
* --simplify &emsp; Simplify the ONNX model
* --dynamic-export &emsp; Export ONNX with dynamic input shapes
* --verify &emsp; Verify the correctness of an exported model by comparing the results with Pytorch
* --show &emsp; Show the graph of ONNX model

For more information and usage details, please refer to [MMClassification official conversion tutorials](https://github.com/open-mmlab/mmclassification/blob/master/docs_zh-CN/tools/pytorch2onnx.md).

### Example: Converting Faster R-CNN

If you do not install MMDetection, following this [tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) to install it first. Faster R-CNN uses some custom operators implemented in `MMCV` which is more efficient than standard ONNX operators. If you want to use these custom operations, you need to build custom operators for ONNX Runtime before installing MMCV. Please refer to the MMCV tutorial [Custom operators for ONNX Runtime in MMCV](https://github.com/open-mmlab/mmcv/blob/master/docs/deployment/onnxruntime_op.md#how-to-build-custom-operators-for-onnx-runtime). If you do not build custom operators, Faster R-CNN will be converted to the ONNX model using standard operations.

Download the corresponding checkpoint, and use the tool to convert the checkpoint to an ONNX model.
```bash
# download the checkpoint from the model zoo and put it in `checkpoints/`
cd mmdetection && mkdir checkpoints && cd checkpoints
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

# converting the checkpoint to an ONNX model
python ../tools/deployment/pytorch2onnx.py ../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
--output-file faster_rcnn.onnx --simplify --dynamic-export
```

The inputs of detection models are usually in different shapes, we recommend using `--dynamic-export` to export the model with dynamic input and output shapes, to ensure the accuracy of the network.
The `faster_rcnn.onnx` will be generated in the current directory. More details can refer to the [MMDetection official converting tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/pytorch2onnx.md).

For a two-stage detector like Faster R-CNN, if the exported ONNX model supports dynamic shapes and multi-batch inputs, MMDetection converting tools fixes the number of proposals to 1000, i.e. the first dimension of Region Proposal Network(RPN) output is 1000. This will introduce unnecessary calculations for input images with less than 1000 proposal boxes. For inference libraries like `PPLNN` that support the `NonZero` operation, the source code of the detector implementation in MMDetection can be slightly modified to avoid the additional calculations mentioned above, and further improve the inference speed.

The ROI extractor in Faster R-CNN is implemented in [mmdetection/mmdet/models/roi_extractors/single_level_roi_extractor.py](https://github.com/open-mmlab/mmdetection/blob/30a7073a328aebccb24bd6bef2f13d3ddfca765f/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py#L85). The original conversion code is in the `forward()` function as follows:

```python
@force_fp32(apply_to=('feats', ), out_fp16=True)
def forward(self, feats, rois, roi_scale_factor=None):
......
for i in range(num_levels):
    mask = target_lvls == i
    if torch.onnx.is_in_onnx_export():
        # To keep all roi_align nodes exported to onnx
        # and skip nonzero op
        mask = mask.float().unsqueeze(-1)
        # select target level rois and reset the rest rois to zero.
        rois_i = rois.clone().detach()
        rois_i *= mask
        mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
        roi_feats_t = self.roi_layers[i](feats[i], rois_i)
        roi_feats_t *= mask_exp
        roi_feats += roi_feats_t
        continue
......
```

To avoid unnecessary calculations, we filter out the proposals with zero values and modify the above code as follows:

```python
@force_fp32(apply_to=('feats', ), out_fp16=True)
def forward(self, feats, rois, roi_scale_factor=None):
......
for i in range(num_levels):
    mask = target_lvls == i
    if torch.onnx.is_in_onnx_export():
        # use nozero to filter out unnecessary proposals
        inds = mask.nonzero(as_tuple=False).squeeze(1)
        rois_ = rois[inds]
        roi_feats_t = self.roi_layers[i](feats[i], rois_)
        roi_feats[inds] = roi_feats_t
        continue
......
```

Then run the converting tools and a simplified ONNX model will be generated.


## Convert Model from PyTorch
PyTorch provides an API called `torch.onnx.export()` to support the model conversion. More information can be get from the [offical docs](https://pytorch.org/docs/stable/onnx.html?highlight=torch%20onnx%20export#torch.onnx.export). Here is a simple example code that exports a pre-trained MobileNetV2 in torchvision into ONNX with dynamic shapes.

```python
import torch
import torchvision

dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
model = torchvision.models.mobilenet_v2(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
input_names = [ "input" ]
output_names = [ "probs" ]

# Providing dynamic axes to export models with dynamic shapes. The dictionary
# specifies a mapping FROM the index of dynamic axis in corresponding
# input/output TO the name that is desired to be applied on such axis
# of such input/output during export.
dynamic_axes = {
    'input': {
        0:'batch',
        2:'width',
        3:'heitht'
        },
    'probs': {
        0: 'batch',
        },
    }

torch.onnx.export(model,
                  dummy_input,
                  "mobilenet_v2.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  do_constant_folding=True,
                  opset_version=11,
                  dynamic_axes=dynamic_axes)
```

PyTorch provides two ways to export a model to the ONNX format, tracing and scripting. The example shown above is trace-based, running an inference and then save the results. The script-based exporter is more friendly to dynamic characteristics. This [documentation](https://pytorch.org/docs/stable/onnx.html) gives more details and some useful examples. You can also follow the tutorial: [Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

PyTorch converter can only provide some simple optimizations, like constant folding. To get a more simplified ONNX model with higher performance, we recommend further simplify the model. ONNX Simplifier is an official recommended tool to simplify the ONNX model. It infers the whole computation graph and then replaces the redundant operators with their constant outputs. The usage guide of ONNX Simplifier can be found in [here](https://github.com/daquexian/onnx-simplifier).
