#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PPDetectionPytorch
# @File    ：convert_paddle_to_torch_base.py
# @Author  ：sl
# @Date    ：2022/11/10 14:14


import os
import paddle

from ppdettorch.utils.model_utils import ModelUtils


class ConvertPaddleToTorchBase(object):

    def __init__(self):
        pass

    @staticmethod
    def load_paddle_model(file_name, show_info=True):
        """
        加载paddle 模型参数

        :param file_name:
        :param show_info:
        :return:
        """
        # paddle_model_params = paddle.load(file_name, return_numpy=True)
        paddle_model_params = ModelUtils.load(file_name, return_numpy=True)

        model_params = []
        for weight_name, weight_value in paddle_model_params.items():
            msg = f"{weight_name} : {weight_value.shape}"
            if show_info:
                print(msg)
            model_params.append(msg)

        return paddle_model_params, model_params
