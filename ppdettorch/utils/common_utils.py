#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/20 - 22:01

from ppdettorch.utils.base_utils import BaseUtil
from ppdettorch.utils.file_utils import FileUtils

"""
通用工具类
"""


class CommonUtils(BaseUtil):
    """
    通用工具类
    """

    def init(self):
        pass

    @staticmethod
    def print_model_param(model, show_model=True, use_numpy=False,
                          show_info=True, save_dir=None,
                          paddle_model=False):
        """
         打印出每一层的参数的大小
        :param model:
        :param show_model:
        :param use_numpy:
        :param show_info:
        :param save_dir:
        :param paddle_model:
        :return:
        """
        if show_info and show_model:
            print(model)

        model_net = str(model)
        model_params = []

        end_type = "torch"
        if paddle_model:
            use_numpy = True
            end_type = "paddle"

        params_dict = model.named_parameters()
        for name, parameters in params_dict:
            parm_size = parameters.size() if not use_numpy else parameters.detach().numpy().shape
            msg = f"{name} : {parm_size}"
            if show_info:
                print(msg)
            model_params.append(msg)

        if save_dir is not None:
            file_name_torch_net = f"{save_dir}_{end_type}_net.txt"
            file_name_torch_param = f"{save_dir}_{end_type}_param.txt"
            FileUtils.save_to_text(file_name_torch_net, model_net)
            FileUtils.save_to_text(file_name_torch_param, "\n".join(model_params))

        return model_net, model_params
