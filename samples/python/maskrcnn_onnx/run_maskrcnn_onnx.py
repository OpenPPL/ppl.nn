import sys
import logging
import cv2
import numpy as np
import argparse
from pyppl import nn as pplnn
from pyppl import common as pplcommon

logging.basicConfig(level=logging.INFO)

coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
        'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
        'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
    ]

class DataProcess(object):
    def __init__(self):
        self._w_scale = 1
        self._h_scale = 1
        self._mean = [123.675, 116.28, 103.53]
        self._std = [58.395, 57.12, 57.375]
        self._data_bin = 'input.bin'
        self._score_thr = 0.5

    def _load_and_resize(self, input_img_path, resized_h, resized_w):
        """Load an image from the specified input path, and resize it to requited shape(h, w)

        Keyword arguments:
        input_img_path -- string path of the image to be loaded
        resized_h -- required shape's h
        resized w -- required shape's w
        """
        self._input_file_name = input_img_path
        img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        self._h_scale = resized_h / img.shape[0]
        self._w_scale = resized_w / img.shape[1]
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        return img

    def _normalize(self, img):
        """ change input img data as a NumPy float array.

        Keyword arguments:
        img -- img's pixel array(numpy array)
        """
        mean = np.array(self._mean).reshape(1, -1).astype(np.float64)
        std = np.array(self._std).reshape(1, -1).astype(np.float64)
        stdinv = 1 / std
        img = img.copy().astype(np.float32)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.subtract(img, mean, img)
        cv2.multiply(img, stdinv, img)
        img = img.transpose(2, 0, 1)
        img.tofile(self._data_bin)

    def preprocess(self, input_img_path, resized_h, resized_w):
        """ change input img as a NumPy float array.

        Keyword arguments:
        input_img_path -- string path of the image to be loaded
        resized_h -- required shape's h
        resized w -- required shape's w
        """
        img = self._load_and_resize(input_img_path, resized_h, resized_w)
        self._normalize(img)
        return self._data_bin

    def _draw_result_to_img(self,
                            boxes_and_score_data,
                            labels_data,
                            masks_data,
                            save_file_name):
        """Draw the bounding boxes and mask on the original input image and save it.

        Keyword arguments:
        boxes_and_score_data -- NumPy array containing the bounding box coordinates of N objects and score, with shape (N,5).
        lables_data -- NumPy array containing the corresponding label for each object, with shape (N,)
        mask_data -- Numpy array containing the mask of N objects
        save_file_name -- out image file name
        """

        im = cv2.imread(self._input_file_name, cv2.IMREAD_COLOR)
        scores = boxes_and_score_data[:, -1]
        inds = scores > self._score_thr
        bboxes = boxes_and_score_data[inds, :]
        labels = labels_data[inds]
        segms = masks_data[inds, ...]

        np.random.seed(42)
        mask_colors = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            bbox_int = bbox.astype(np.int32)
            left = int(bbox_int[0] / self._w_scale)
            top = int(bbox_int[1] / self._h_scale)
            right = int(bbox_int[2] / self._w_scale)
            bottom = int(bbox_int[3] / self._h_scale)
            cv2.rectangle(im, (left, top), (right, bottom), (0, 0, 0), 2)
            cv2.putText(im,
                        coco_classes[label] + ": " + str(round(bbox[4], 2)),
                        (left, top),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1.0,
                        (0, 0, 0))
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i]
                mask = cv2.resize(mask, (im.shape[1], im.shape[0]))
                mask = mask.astype(bool)
                im[mask] = im[mask] * 0.5 + color_mask * 0.5
        cv2.imwrite(save_file_name, im)

    def postprocess(self, boxes_and_score_data, labels_data, masks_data, out_file_name):
        self._draw_result_to_img(boxes_and_score_data, labels_data, masks_data, out_file_name)

class PPLModel(object):
    def __init__(self):
        self._engines = []

    def _create_x86_engine(self):
        x86_options = pplnn.X86EngineOptions()
        x86_engine = pplnn.X86EngineFactory.Create(x86_options)
        self._engines.append(pplnn.Engine(x86_engine))

    def _create_cuda_engine(self):
        cuda_options = pplnn.CudaEngineOptions()
        cuda_options.device_id = 0
        cuda_engine = pplnn.CudaEngineFactory.Create(cuda_options)
        self._engines.append(pplnn.Engine(cuda_engine))

    def _create_runtime(self, model_file_name):
        runtime_builder = pplnn.OnnxRuntimeBuilderFactory.CreateFromFile(model_file_name, self._engines)
        if not runtime_builder:
            logging.error("create RuntimeBuilder failed.")
            sys.exit(-1)

        self._runtime = runtime_builder.CreateRuntime()
        if not self._runtime:
            logging.error("create Runtime instance failed.")
            sys.exit(-1)

    def _prepare_input(self, input_file):
        """  set input data
        """
        tensor = self._runtime.GetInputTensor(0)
        in_data = np.fromfile(input_file, dtype=np.float32).reshape((1, 3, 800, 1200))
        status = tensor.ConvertFromHost(in_data)
        if status != pplcommon.RC_SUCCESS:
            logging.error("copy data to tensor[" + tensor.GetName() + "] failed: " +
                          pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    def _prepare_output(self):
        """ save output
        """
        for i in range(self._runtime.GetOutputCount()):
            tensor = self._runtime.GetOutputTensor(i)
            tensor_data = tensor.ConvertToHost()
            if not tensor_data:
                logging.error("copy data from tensor[" + tensor.GetName() + "] failed.")
                sys.exit(-1)
            if tensor.GetName() == 'dets':
                dets_data = np.array(tensor_data, copy=False)
                dets_data = dets_data.squeeze()
            if tensor.GetName() == 'labels':
                labels_data = np.array(tensor_data, copy=False)
                labels_data = labels_data.squeeze()
            if tensor.GetName() == 'masks':
                masks_data = np.array(tensor_data, copy=False)
                masks_data = masks_data.squeeze()
        return dets_data, labels_data, masks_data

    def run(self, engine_type, model_file_name, input_file):
        """ run pplmodel

        Keyword arguments:
        engine_type -- which engine to use x86 or cuda
        model_file_name -- input model file
        input_file -- input data file (binary data)
        """
        if engine_type == 'x86':
            self._create_x86_engine()
        elif engine_type == 'cuda':
            self._create_cuda_engine()
        else:
            logging.error('not support engine type: ', engine_type)
            sys.exit(-1)
        self._create_runtime(model_file_name)
        self._prepare_input(input_file)
        status = self._runtime.Run()
        if status != pplcommon.RC_SUCCESS:
            logging.error("Run() failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)
        dets_data, labels_data, masks_data = self._prepare_output()
        return dets_data, labels_data, masks_data

def parsArgs():
    parser = argparse.ArgumentParser("runner of maskrcnn onnx model.")
    parser.add_argument('-i', '--in_img_file', type=str, dest='in_img_file', required=True, help="Specify the input image.")
    parser.add_argument('-o', '--out_img_file', type=str, dest='out_img_file', required=False, help="Specify the output image's name.")
    parser.add_argument('-m', '--onnx_model', type=str, dest='onnx_model', required=True, help="Specify the onnx model path.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parsArgs()
    if args.in_img_file is None:
        logging.error("no input img file was select.")
        sys.exit(1)

    if args.out_img_file is None:
        args.out_img_file = 'test_out.jpg'
        logging.info("out img name was not select, use default name [test_out.jpg].")

    if args.onnx_model is None:
        logging.error('onnx model was not select.')
        sys.exit(1)

    # prepare data
    image_file = args.in_img_file
    result_file = args.out_img_file
    model_file = args.onnx_model

    data_process = DataProcess()
    # preprocess
    input_data = data_process.preprocess(image_file, 800, 1200)

    # runmodel
    ppl_model = PPLModel()
    dets_data, labels_data, masks_data = ppl_model.run('x86', model_file, input_data)

    # postprocess
    data_process.postprocess(dets_data, labels_data, masks_data, result_file)

    logging.info("Run ok")
