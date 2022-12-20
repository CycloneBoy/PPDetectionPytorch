#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/20 - 21:58
import os
from abc import abstractmethod, ABC
from enum import Enum, unique


class BaseUtil(ABC):
    """
    抽取数据基类
    """

    @abstractmethod
    def init(self):
        """
        工具类初始化
        :return:
        """
        pass
