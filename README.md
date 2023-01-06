# PPDetectionPytorch

> use pytorch to reimplement paddle detection

[简体中文](README_cn.md) | [English](README_en.md)

## illustrate
This project aims to:
- Learn object detection and PPDetection code
- Enable PPDetection model to be used on pytorch
- Provide a reference for paddle to pytorch

## TODO
- [x] Coordinate transformation of target detection in yolov3 inference part
- [ ] Rotating frame detection model PP-YOLOE-R
- [ ] Small target detection model PP-YOLOE-SOD
- [ ] PP-Human v2
- [ ] PP-Vehicle
- [ ] Object detection model training part
- [ ] Multiple target tracking
- [ ] ...

## Notice

`PPDetectionPytorch` is a pytorch implementation based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) and [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO).

**RECENT UPDATED**

- 2023-01-06 Coordinate transformation of target detection in yolov3 inference part
- 2022-12-21 Migration model inference part: picodet, ppyoloe, yolox, yolov5, yolov6, yolov7, rtmdet
- 2022-12-19 Base code migration

## Instructions for use

### Code path modification

[WORK_DIR](./ppdettorch/utils/constant.py)  Constants.WORK_DIR is changed to the absolute path of your own code

### Model Inference
#### Model Weights
> Automatically download the corresponding model weights in configs to the ~/.cache/paddle/weights directory during the first inference. This directory is the specified directory for downloading the PPDetection inference model.
> Model weights will be automatically converted to pytoch weights and saved to the current directory.

[Model weight automatic conversion code](./ppdettorch/process/transform/convert_paddle_detection_to_torch.py)
#### Run inference tests
> Currently only part of the model weights are converted:
The run_picodet_coco_batch function in [run_detection_infer.py](./tests/process/infer/run_detection_infer.py) uses different model_classes for reasoning tests
>

#### Inference result output
> Output inference results and corresponding network paddle and pytorch model weight names
[Inference output directory](./outputs/models/detection)


## refer to

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)
- [Paddle](https://github.com/PaddlePaddle)
- [Pytorch](https://pytorch.org/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)