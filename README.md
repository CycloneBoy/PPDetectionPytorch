# PPDetectionPytorch

> use pytorch to reimplement paddle detection

[简体中文](README_cn.md) | [English](README_en.md)

## 说明
本项目旨在：
- 学习目标检测和PPDetection代码
- 使得PPDetection的模型能在pytorch上使用
- 提供为paddle转pytorch提供参考

## TODO
- [ ] yolov3 推理部分的目标检测况坐标转换部分
- [ ] 旋转框检测模型PP-YOLOE-R 
- [ ] 小目标检测模型PP-YOLOE-SOD 
- [ ] PP-Human v2 
- [ ] PP-Vehicle
- [ ] 目标检测模型训练部分
- [ ] 多目标跟踪
- [ ] ...

## 注意

`PPDetectionPytorch`是基于[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 和 [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO) 的pytorch 实现。

**近期更新**

- 2022-12-21 迁移模型推理部分：picodet,ppyoloe,yolox,yolov5,yolov6,yolov7,rtmdet
- 2022-12-19 基础代码迁移

## 使用说明

### 代码路径修改

[WORK_DIR](./ppdettorch/utils/constant.py)  Constants.WORK_DIR 改为你自己的代码绝对路径

### 依赖
- paddlepaddle-gpu 需要安装 ，这里用于自动将PPDetection提供的模型权重转换成pytorch权重
- pytorch 需要安装

### 模型推理
#### 模型权重
> 首次推理时自动下载对应configs 中的模型权重到 ~/.cache/paddle/weights目录，这个目录是PPDetection推理模型的下载指定目录，
> 模型权重会自动转换成pytoch 权重，同时保存到当前目录。
[模型权重自动转换代码](./convert_paddle_detection_to_torch.py)
#### 运行推理测试
> 目前只转换了部分模型权重: 
[run_detection_infer.py](./tests/process/infer/run_detection_infer.py) 中 run_picodet_coco_batch函数 使用不同的model_class 进行推理测试
> 

#### 推理结果输出
> 输出推理结果和对应网络的paddle和pytorch模型权重名称
[推理输出目录](./outputs/models/detection)



## 参考

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)
- [Paddle](https://github.com/PaddlePaddle)
- [Pytorch](https://pytorch.org/)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)