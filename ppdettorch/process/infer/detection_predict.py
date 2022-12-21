#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ： 
# @File    ：detection_predict.py
# @Author  ：sl
# @Date    ：2022/11/7 17:35


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import glob

import torch
from ppdettorch.utils.detection_common_utils import DetectionCommonUtils

from ppdettorch.engine.detection_trainer import DetectionTrainer
from ppdettorch.utils.detection_infer_utils import DetectionInferUtils
from ppdettorch.utils.file_utils import FileUtils

from ppdettorch.utils.logger_utils import logger
from ppdettorch.core.workspace import load_config, merge_config
from ppdettorch.utils.detection_cli_utils import ArgsParser, merge_args


"""
Detection 模型推理
"""


def run(args, cfg):
    # build trainer
    trainer = DetectionTrainer(cfg, mode='test')

    # load weights
    weight_output_dir = f"{args.output_dir}/network/{FileUtils.get_file_name(args.config)}"
    trainer.load_weights(cfg.weights, do_transform=args.do_transform, output_dir=weight_output_dir)

    # get inference images
    images = DetectionInferUtils.get_test_images(args.infer_dir, args.infer_img)

    # inference
    if args.slice_infer:
        trainer.slice_predict(
            images,
            slice_size=args.slice_size,
            overlap_ratio=args.overlap_ratio,
            combine_method=args.combine_method,
            match_threshold=args.match_threshold,
            match_metric=args.match_metric,
            draw_threshold=args.draw_threshold,
            output_dir=args.output_dir,
            save_results=args.save_results,
            visualize=args.visualize)
    else:
        trainer.predict(
            images,
            draw_threshold=args.draw_threshold,
            output_dir=args.output_dir,
            save_results=args.save_results,
            visualize=args.visualize)


def main(args):
    cfg = load_config(args.config)
    merge_args(cfg, args)
    merge_config(args.opt)

    DetectionCommonUtils.check_config(cfg)

    run(args, cfg)


if __name__ == '__main__':
    run_arg = DetectionInferUtils.init_args()
    main(run_arg)
