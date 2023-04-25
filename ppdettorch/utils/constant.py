#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/20 - 21:56
import os
import re


class Constants(object):
    """
    常量工具类

    """
    USER_HOME = os.path.expanduser("~")
    WORK_DIR = f"{USER_HOME}/work/PPDetectionPytorch"
    # WORK_DIR = "./"

    DATA_DIR = f"{WORK_DIR}/data"
    LOG_DIR = f"{WORK_DIR}/logs"

    # 日志相关
    LOG_FILE = f"{LOG_DIR}/run.log"
    LOG_LEVEL = "debug"

    # models
    OUTPUT_MODELS_DIR = f"{WORK_DIR}/outputs/models"
