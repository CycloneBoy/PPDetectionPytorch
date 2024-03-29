#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：ppdettorch
# @File    ：run_detection_infer.py
# @Author  ：sl
# @Date    ：2022/11/7 18:23
from ppdettorch.utils.logger_utils import logger

from ppdettorch.utils.detection_infer_utils import DetectionInferUtils
from ppdettorch.utils.file_utils import FileUtils
from ppdettorch.utils.time_utils import TimeUtils

from ppdettorch.process.infer.detection_predict import main as main_detection

from ppdettorch.utils.constant import Constants

"""
检测 推理
https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_enhance/picodet_s_192_lcnet_pedestrian.pdparams
"""


class DetectionRunInfer(object):

    def __init__(self):
        self.base_dir_image = f"{Constants.DATA_DIR}/ocr/imgs_words/ch"
        self.run_time = TimeUtils.now_str_short()
        self.base_dir = Constants.WORK_DIR
        self.detection_config_dir = f"{Constants.WORK_DIR}/configs"
        self.checkpoint_base_url = "https://paddledet.bj.bcebos.com/models"
        self.checkpoint_base_url_paddledet = "https://bj.bcebos.com/v1/paddledet/models"
        self.checkpoint_base_url_ppstructure = "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/"
        self.checkpoint_base_url_pedestrian = "https://bj.bcebos.com/v1/paddledet/models/keypoint/tinypose_enhance/"
        # self.checkpoint_base_url = "~/.cache/paddle/weights/"

    def run_picodet_coco_model(self, config_file=None, do_transform=True):
        """
            测试 picodet

        :param config_file:
        :param do_transform:
        :return:
        """
        # config_file = f"{self.base_dir}/configs/picodet/{config_name}.yml"
        model_name = FileUtils.get_file_name(config_file)

        layout_zh = False

        predict_labels = None
        if model_name in ["picodet_lcnet_x1_0_layout", ]:
            layout_zh = True
            base_url = self.checkpoint_base_url_ppstructure
            predict_labels = "~/PaddleOCR/ppocr/utils/dict/layout_dict/layout_cdla_dict.txt"
        elif model_name in ["picodet_s_192_pedestrian", "picodet_s_320_pedestrian",
                            "picodet_s_192_lcnet_pedestrian", "picodet_s_320_lcnet_pedestrian"]:
            base_url = self.checkpoint_base_url_pedestrian
            # model_name = str(model_name).replace("_pedestrian", "_pedestrian")
        elif model_name in ['retinanet_r50_fpn_2x_coco']:
            base_url = self.checkpoint_base_url
            model_name = "retinanet_r101_distill_r50_2x_coco"
        elif model_name in ["rtdetr_r34vd_6x_coco", "rtdetr_r18vd_6x_coco"]:
            base_url = self.checkpoint_base_url_paddledet
        else:
            base_url = self.checkpoint_base_url

        # model param name replace
        model_params_config = {
            "rtdetr_r34vd_6x_coco": "rtdetr_r34vd_dec4_6x_coco",
            "rtdetr_r18vd_6x_coco": "rtdetr_r18vd_dec3_6x_coco",
        }

        if layout_zh:
            checkpoint_file = f"{base_url}/picodet_lcnet_x1_0_fgd_layout_cdla.pdparams"
        elif model_name in model_params_config.keys():
            checkpoint_file = f"{base_url}/{model_params_config.get(model_name, model_name)}.pdparams"
        else:
            checkpoint_file = f"{base_url}/{model_name}.pdparams"

        # picodet_lcnet_x1_0_fgd_layout_cdla.pdparams
        # picodet_lcnet_x1_0_fgd_layout.pdparams
        # checkpoint_file = f"{self.checkpoint_base_url}/{config_name}.pth"

        run_arg = DetectionInferUtils.init_args()
        run_arg.config = config_file
        run_arg.opt = {
            "use_gpu": True,
            "weights": checkpoint_file
        }

        if layout_zh:
            run_arg.opt["num_classes"] = 10

        run_arg.infer_img = f"{self.base_dir}/demo/000000014439.jpg"
        # run_arg.infer_img = f"{self.base_dir}/demo/car.jpg"
        # run_arg.infer_img = f"{self.base_dir}/docs/images/layout.jpg"
        # run_arg.infer_img = f"{self.base_dir}/docs/images/layout_demo2.png"
        # run_arg.infer_img = f"{Constants.USER_HOME}/ocr/PaddleOCR/ppstructure/docs/table/layout_demo2.png"
        run_arg.predict_labels = predict_labels

        model_class = self.get_model_class(config_file)
        run_arg.output_dir = f"{Constants.OUTPUT_MODELS_DIR}/detection/{model_class}/{model_name}/inference_results/{self.run_time}"

        run_arg.do_transform = do_transform
        main_detection(run_arg)

    def get_model_class(self, config_file):
        begin_index = len(f"{self.detection_config_dir}/")
        config_name = config_file[begin_index:]
        end_index = config_name.find("/")
        model_class = config_name[:end_index]
        return model_class

    def run_picodet_coco(self, config_name=None):
        """
        测试 picodet

        :return:
        """
        # picodet
        config_name = "picodet_s_320_coco"
        config_name = "picodet_s_416_coco"
        config_name = "picodet_m_320_coco"
        config_name = "picodet_m_416_coco"
        config_name = "picodet_l_320_coco"
        config_name = "picodet_l_416_coco"
        config_name = "picodet_l_640_coco"

        config_name = "/more_config/picodet_lcnet_1_5x_416_coco"
        config_name = "/more_config/picodet_lcnet_1_5x_640_coco"
        config_name = "/more_config/picodet_shufflenetv2_1x_416_coco"
        config_name = "/more_config/picodet_mobilenetv3_large_1x_416_coco.yml"
        config_name = "/more_config/picodet_r18_640_coco.yml"

        config_name = "/application/layout_analysis/picodet_lcnet_x1_0_layout.yml"
        # config_name = "/application/layout_analysis/picodet_lcnet_x2_5_layout.yml"
        # config_name = "/application/mainbody_detection/picodet_lcnet_x2_5_640_mainbody.yml"

        # config_name = "/application/pedestrian_detection/picodet_s_192_pedestrian.yml"
        # config_name = "/application/pedestrian_detection/picodet_s_320_pedestrian.yml"
        # config_name = "/application/pedestrian_detection/picodet_s_192_lcnet_pedestrian.yml"
        # config_name = "/application/pedestrian_detection/picodet_s_320_lcnet_pedestrian.yml"

        # ppyoloe
        # config_name = "ppyoloe_crn_s_300e_coco.yml"

        # yolox
        # config_name = "yolox_s_300e_coco.yml"

        # yolov3
        # config_name = "yolov3_mobilenet_v1_270e_coco.yml"
        # config_name = "yolov3_mobilenet_v3_large_270e_coco.yml"
        # config_name = "yolov3_darknet53_270e_coco.yml"

        # # yolov5
        # config_name = "yolov5_s_300e_coco.yml"
        #
        # # convnext
        # config_name = "yolov5_convnext_s_36e_coco.yml"
        # config_name = "yolox_convnext_s_36e_coco.yml"
        # config_name = "ppyoloe_convnext_tiny_36e_coco.yml"

        # # yolov6
        # config_name = "yolov6_n_300e_coco.yml"
        # config_name = "yolov6_s_300e_coco.yml"
        # config_name = "yolov6_m_300e_coco.yml"
        config_name = "yolov6_l_300e_coco.yml"

        # config_name = "yolov6lite/yolov6lite_s_400e_coco.yml"
        # config_name = "yolov6lite/yolov6lite_m_400e_coco.yml"
        # config_name = "yolov6lite/yolov6lite_l_400e_coco.yml"

        # # yolov7
        # config_name = "yolov7_l_300e_coco.yml"
        #
        # # rtmdet
        # config_name = "rtmdet_s_300e_coco.yml"

        # yolov8
        # config_name = f"yolov8_n_500e_coco.yml"

        # ssd
        # config_name = f"ssd_mobilenet_v1_300_120e_voc.yml"
        # config_name = f"ssd_vgg16_300_240e_voc.yml"
        # config_name = f"ssdlite_mobilenet_v3_large_320_coco.yml"
        # config_name = f"ssdlite_mobilenet_v3_small_320_coco.yml"

        # centernet
        # config_name = f"centernet_mbv1_140e_coco.yml"

        # blazeface
        # config_name = f"blazeface_1000e.yml"
        # config_name = f"blazeface_fpn_ssh_1000e.yml"

        # retinanet
        # config_name = f"retinanet_r50_fpn_1x_coco.yml"
        # config_name = f"retinanet_r101_fpn_2x_coco.yml"
        # config_name = f"retinanet_r50_fpn_2x_coco.yml"

        # rtdetr
        # config_name = f"rtdetr_r50vd_6x_coco.yml"
        # config_name = f"rtdetr_r101vd_6x_coco.yml"
        # config_name = f"rtdetr_hgnetv2_l_6x_coco.yml"
        # config_name = f"rtdetr_hgnetv2_x_6x_coco.yml"
        # config_name = f"rtdetr_r50vd_m_6x_coco.yml"
        # config_name = f"rtdetr_r34vd_6x_coco.yml"
        # config_name = f"rtdetr_r18vd_6x_coco.yml"

        # run_arg = DetectionInferUtils.init_args()
        config_name = config_name if not config_name.endswith(".yml") else config_name[:-4]
        # config_file = f"{self.base_dir}/configs/picodet/legacy_model/{config_name}.yml"

        if "convnext" in config_name:
            model_class = "convnext"
        elif "picodet" in config_name:
            model_class = "picodet"
        elif "ppyoloe" in config_name:
            model_class = "ppyoloe"
        elif "ssdlite_" in config_name:
            model_class = "ssd"
        elif "blazeface_" in config_name:
            model_class = "face_detection"
        elif "yolov6lite_" in config_name:
            model_class = "yolov6"
        else:
            config_name_end_index = FileUtils.get_file_name(config_name).find("_")
            model_class = config_name[:config_name_end_index]

        config_file = f"{self.detection_config_dir}/{model_class}/{config_name}.yml"

        self.run_picodet_coco_model(config_file=config_file)

    def run_picodet_coco_batch(self):
        """
        批量处理
            picodet_l_320_coco_lcnet.yml
        :return:
        """
        base_dir = f"{self.detection_config_dir}/picodet"
        file_name_list = FileUtils.list_dir_or_file(file_dir=base_dir,
                                                    add_parent=True,
                                                    sort=True,
                                                    is_dir=False,
                                                    start_with="picodet_",
                                                    end_with="_lcnet.yml", )

        logger.info(f"total: {len(file_name_list)}")
        skip = 7
        for index, file_name in enumerate(file_name_list):
            if index < skip:
                logger.info(f"跳过已经执行的：{index} - {file_name}")
                continue
            logger.info(f"开始执行：{index} - {file_name}")

            self.run_picodet_coco_model(config_file=file_name)


def demo_run_detection_infer():
    detection_runner = DetectionRunInfer()
    detection_runner.run_picodet_coco()
    # detection_runner.run_picodet_coco_batch()


def run_picodet_coco_batch():
    """
    批量处理

    :return:
    """
    # model_class = "picodet"
    # model_class = "ppyoloe"
    # model_class = "yolox"
    # model_class = "yolov3"
    # model_class = "yolov5"
    # model_class = "yolov6"
    # model_class = "yolov7"
    # model_class = "rtmdet"
    # model_class = "yolov8"
    # model_class = "ssd"
    model_class = "rtdetr"

    with_application = False
    # with_application = True

    # do_transform = False
    do_transform = True

    # 需要跳过执行验证的列表
    skip_config_name_dict = {
        "yolov3": [
            "yolov3_darknet53_original_270e_coco.yml",
            "yolov3_mobilenet_v1_roadsign.yml"
        ],
        "ssd": [
            "ssd_r34_70e_coco.yml",
            "ssdlite_ghostnet_320_coco.yml"
        ]
    }

    base_dir = f"{Constants.WORK_DIR}/configs/{model_class}"
    if with_application:
        base_dir = f"{base_dir}/application"
    file_name_list = FileUtils.list_dir_or_file(file_dir=base_dir,
                                                add_parent=True,
                                                sort=True,
                                                is_dir=False,
                                                start_with=f"{model_class}",
                                                end_with=".yml", )

    logger.info(f"total: {len(file_name_list)}")

    skip_config_name_list = skip_config_name_dict.get(model_class, [])
    skip = 1
    detection_runner = DetectionRunInfer()

    for index, file_name in enumerate(file_name_list):
        if index < skip:
            logger.info(f"跳过已经执行的：{index} - {file_name}")
            continue

        if f"{FileUtils.get_file_name(file_name)}.yml" in skip_config_name_list:
            logger.info(f"跳过无需测试的：{index} - {file_name}")
            continue

        if "_xpu" in file_name:
            continue
        if "ppyoloe_crn_m_80e_pcb" in file_name:
            continue
        logger.info(f"开始执行：{index} - {file_name}")

        detection_runner.run_picodet_coco_model(config_file=file_name, do_transform=do_transform)

    logger.info(f"完成所有检测: {len(file_name_list)}")


if __name__ == '__main__':
    demo_run_detection_infer()
    # run_picodet_coco_batch()
