#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/20 - 22:06
import re

from ppdettorch.utils.base_utils import BaseUtil

"""
匹配工具类类

"""


class MatchUtils(BaseUtil):
    """
    匹配工具类类

    """

    # paddle to torch 0.0.
    PATTERN_PADDLE_NUMBER_REPLACE = re.compile(r"(.*)(\d+)\.(\d+)\.(.*)")
    # layers1.stage1.conv_layer layers(\d+)\.stage(\d+)\.conv_layer
    PATTERN_PADDLE_CSP_DARKNET_REPLACE = re.compile(r"(.*)layers(\d+)\.stage(\d+)\.(.*)")

    @staticmethod
    def match_pattern_extract(text, pattern, return_str=False):
        """
        匹配单个位置
        :param text:
        :param pattern:
        :param return_str:
        :return:
        """
        if isinstance(pattern, str):
            match_pattern = re.compile(pattern)
        else:
            match_pattern = pattern

        match_result = match_pattern.findall(text)
        if len(match_result) > 0:
            if return_str:
                res = str(match_result[0]).rstrip()
            else:
                res = match_result
        else:
            res = ""
        return res

    @staticmethod
    def match_pattern_flag(text, pattern):
        """
        匹配结果
        :param text:
        :param pattern:
        :return:
        """
        flag = False
        filter_error_type = len(MatchUtils.match_pattern_extract(text, pattern)) > 0
        if filter_error_type:
            flag = True
        return flag

    @staticmethod
    def match_pattern_list(texts, pattern_list):
        """
        匹配  正则列表
        """
        if not isinstance(pattern_list, list):
            pattern_list = [pattern_list]

        match_result = []
        for pattern in pattern_list:
            raw_match_result = pattern.findall(texts)
            if len(raw_match_result) > 0:
                match_result = raw_match_result
                break

        return match_result


    @staticmethod
    def match_pattern_list_flag(texts, pattern_list):
        """
        匹配  正则列表 FLAG
        """
        match_flag = False
        match_result = MatchUtils.match_pattern_list(texts=texts, pattern_list=pattern_list)
        if len(match_result) > 0:
            match_flag = True

        return match_flag