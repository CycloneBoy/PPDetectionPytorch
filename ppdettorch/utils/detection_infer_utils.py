#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：financial_ner 
# @File    ：detection_infer_utils.py
# @Author  ：sl
# @Date    ：2022/11/7 18:27
import ast
import glob
import os

from ppdettorch.utils.logger_utils import logger

from ppdettorch.utils.base_utils import BaseUtil
from ppdettorch.utils.detection_cli_utils import ArgsParser

"""
 Detection 推理工具
"""


class DetectionInferUtils(BaseUtil):

    def init(self):
        pass

    @staticmethod
    def init_args():
        parser = ArgsParser()
        parser.add_argument(
            "--infer_dir",
            type=str,
            default=None,
            help="Directory for images to perform inference on.")
        parser.add_argument(
            "--infer_img",
            type=str,
            default=None,
            help="Image path, has higher priority over --infer_dir")
        parser.add_argument(
            "--output_dir",
            type=str,
            default="output",
            help="Directory for storing the output visualization files.")
        parser.add_argument(
            "--draw_threshold",
            type=float,
            default=0.5,
            help="Threshold to reserve the result for visualization.")
        parser.add_argument(
            "--slim_config",
            default=None,
            type=str,
            help="Configuration file of slim method.")
        parser.add_argument(
            "--use_vdl",
            type=bool,
            default=False,
            help="Whether to record the data to VisualDL.")
        parser.add_argument(
            '--vdl_log_dir',
            type=str,
            default="vdl_log_dir/image",
            help='VisualDL logging directory for image.')
        parser.add_argument(
            "--save_results",
            type=bool,
            default=False,
            help="Whether to save inference results to output_dir.")
        parser.add_argument(
            "--slice_infer",
            action='store_true',
            help="Whether to slice the image and merge the inference results for small object detection."
        )
        parser.add_argument(
            '--slice_size',
            nargs='+',
            type=int,
            default=[640, 640],
            help="Height of the sliced image.")
        parser.add_argument(
            "--overlap_ratio",
            nargs='+',
            type=float,
            default=[0.25, 0.25],
            help="Overlap height ratio of the sliced image.")
        parser.add_argument(
            "--combine_method",
            type=str,
            default='nms',
            help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
        )
        parser.add_argument(
            "--match_threshold",
            type=float,
            default=0.6,
            help="Combine method matching threshold.")
        parser.add_argument(
            "--match_metric",
            type=str,
            default='ios',
            help="Combine method matching metric, choose in ['iou', 'ios'].")
        parser.add_argument(
            "--visualize",
            type=ast.literal_eval,
            default=True,
            help="Whether to save visualize results to output_dir.")
        parser.add_argument(
            "--do_transform",
            type=bool,
            default=True,
            help="Whether to transform paddle model to pytorch .")
        parser.add_argument(
            "--predict_labels",
            type=str,
            default=None,
            help="predict labels file")

        args = parser.parse_args()
        return args

    @staticmethod
    def get_test_images(infer_dir, infer_img):
        """
        Get image path list in TEST mode
        """
        assert infer_img is not None or infer_dir is not None, \
            "--infer_img or --infer_dir should be set"
        assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
        assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

        # infer_img has a higher priority
        if infer_img and os.path.isfile(infer_img):
            return [infer_img]

        images = set()
        infer_dir = os.path.abspath(infer_dir)
        assert os.path.isdir(infer_dir), \
            "infer_dir {} is not a directory".format(infer_dir)
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
        images = list(images)

        assert len(images) > 0, "no image found in {}".format(infer_dir)
        logger.info("Found {} inference images in total.".format(len(images)))

        return images
