#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PPDetectionPytorch
# @File    ：detection_common_utils.py
# @Author  ：sl
# @Date    ：2022/11/3 16:36
import sys

from .base_utils import BaseUtil
from .logger_utils import logger

"""
detection 通用工具类
"""


class DetectionCommonUtils(BaseUtil):
    """
    通用工具类
    """

    def init(self):
        pass

    @staticmethod
    def get_unique_endpoints(trainer_endpoints):
        # Sorting is to avoid different environmental variables for each card
        trainer_endpoints.sort()
        ips = set()
        unique_endpoints = set()
        for endpoint in trainer_endpoints:
            ip = endpoint.split(":")[0]
            if ip in ips:
                continue
            ips.add(ip)
            unique_endpoints.add(endpoint)
        logger.info("unique_endpoints {}".format(unique_endpoints))
        return unique_endpoints

    @staticmethod
    def check_config(cfg):
        """
        Check the correctness of the configuration file. Log error and exit
        when Config is not compliant.
        """
        err = "'{}' not specified in config file. Please set it in config file."
        check_list = ['architecture', 'num_classes']
        try:
            for var in check_list:
                if not var in cfg:
                    logger.error(err.format(var))
                    sys.exit(1)
        except Exception as e:
            pass

        if 'log_iter' not in cfg:
            cfg.log_iter = 20

        return cfg
