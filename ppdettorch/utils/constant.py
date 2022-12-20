#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/20 - 21:56


import re


class Constants(object):
    """
    常量工具类

    """
    WORK_DIR = "./"

    DATA_DIR = f"{WORK_DIR}/data"
    LOG_DIR = f"{WORK_DIR}/logs"

    # 日志相关
    LOG_FILE = f"{LOG_DIR}/run.log"
    LOG_LEVEL = "debug"
