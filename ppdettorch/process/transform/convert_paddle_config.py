#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PPDetectionPytorch
# @File    ：convert_paddle_config.py
# @Author  ：sl
# @Date    ：2023/1/20 11:52
import re

CONVERTER_PADDLE_MODEL_CONFIG = {
    "yolov3": [
        re.compile(r"(.*)\.(\d+)\.(\d+)\.(.*)"),
        re.compile(r"(.*)\.(\d+)\.(downsample)\.(.*)"),
        re.compile(r"(.*)\.(\d+)\.(.*)")
    ],
    "DB": [
        re.compile(r"(backbone\.stage)(\d+)\.(\d+)\.(.*)"),
    ],
}

CONVERTER_PADDLE_MODEL_WEIGHT_CONFIG = {
    "SLANet": [
        re.compile(r"structure_attention_cell.(i2h|h2h|score).weight"),
        re.compile(r"(structure_generator|loc_generator)\.(\d+)\.weight"),
    ],
    # 语言模型
    "LM": [
        re.compile(r"encoder.layer\.(\d+)\..*\.weight"),
        re.compile(r"visual_proj.weight"),
    ]
}
# 模型参数变换
MODEL_PARAMS_CONFIG = {
    "yolov3": {
        "rename": [
            re.compile(r"(.*)\.(\d+)\.(\d+)\.(.*)"),
            re.compile(r"(.*)\.(\d+)\.(downsample)\.(.*)"),
            re.compile(r"(.*)\.(\d+)\.(.*)")
        ],
        "transpose": [],
        "filter": [],
    },
    "yolov5": {
        "rename": [
            re.compile(r"(yolo_head\.yolo_output)\.(\d+)\.(.*)")
        ],
        "transpose": [
            # yolov5_convnext_s
            re.compile(r"backbone.stages.(\d+).(\d+).pwconv(\d+).weight")
        ],
        "filter": [],
    },
    "yolov5p6": {
        "rename": [
            re.compile(r"(yolo_head\.yolo_output)\.(\d+)\.(.*)")
        ],
        "transpose": [],
        "filter": [],
    },
    "yolov6": {
        "rename": [
            re.compile(r"(backbone.stage\d+)\.(repconv|replayer|simsppf|bepc3layer|sppf)(.*)")
        ],
        "transpose": [],
        "filter": [],
    },
    "yolov7": {
        "rename": [
            re.compile(r"(backbone.layers\d+)\.(stage\d+)\.(.*)"),
            re.compile(r"(yolo_head.yolo_output)\.(\d+)(.*)")
        ],
        "transpose": [],
        "filter": [],
    },
    "yolov7p6": {
        "rename": [
            re.compile(r"(backbone.layers\d+)\.(stage\d+)\.(.*)"),
            re.compile(r"(yolo_head.yolo_output)\.(\d+)(.*)")
        ],
        "transpose": [],
        "filter": [],
    },
    "yolox": {
        "rename": [],
        "transpose": [
            # yolox_convnext_s_36e_coco
            re.compile(r"backbone.stages.(\d+).(\d+).pwconv(\d+).weight")
        ],
        "filter": [],
    },
    "ppyoloe": {
        "rename": [],
        "transpose": [
            # ppyoloe_convnext_tiny_36e_coco
            re.compile(r"backbone.stages.(\d+).(\d+).pwconv(\d+).weight")
        ],
        "filter": [],
    },
    "rtmdet": {
        "rename": [
            re.compile(r"(backbone.layers\d+)\.(stage\d+)\.(.*)"),
        ],
        "transpose": [],
        "filter": [],
    },
    "yolov8": {
        "rename": [
            re.compile(r"(backbone.layers\d+)\.(stage\d+)\.(.*)"),
            # re.compile(r"(yolo_head.yolo_output)\.(\d+)(.*)")
        ],
        "transpose": [],
        "filter": [],
        "add": ["yolo_head.bce.pos_weight"],
    },
    "DB": {
        "rename": [
            re.compile(r"(backbone\.stage)(\d+)\.(\d+)\.(.*)"),
        ],
        "transpose": [],
        "filter": [],
    },
    "SLANet": {
        "rename": [],
        "transpose": [
            re.compile(r"structure_attention_cell.(i2h|h2h|score).weight"),
            re.compile(r"(structure_generator|loc_generator)\.(\d+)\.weight"),
        ],
        "filter": [],
    },
    # 语言模型
    "LM": {
        "rename": [],
        "transpose": [
            re.compile(r"encoder.layer(s)*\.(\d+)\..*\.weight"),
            re.compile(r"visual_proj.weight"),
            re.compile(r"classifier\.weight"),
        ],
        "filter": [],
    },
    # OCR - DET
    "CRNN": {
        "rename": [
            re.compile(r"(Student\.)(.*)"),
        ],
        "transpose": [
            re.compile(r"head\.ctc_encoder\.encoder\.svtr_block\.(\d+)\..*\.weight"),
            re.compile(r"head\.ctc_head\.fc\.weight"),
            re.compile(r"head\.sar_head\.decoder\.(conv1x1_2|prediction)\.weight"),
        ],
        "filter": [
            re.compile(r"Teacher\..*"),
            re.compile(r"head.sar_head.(encoder|decoder).rnn_(encoder|decoder).(\d+).cell.*"),
        ],
    },
    # OCR - CLS
    "CLS": {
        "rename": [],
        "transpose": [
            re.compile(r"head\.fc\.weight"),
        ],
        "filter": [],
    },
    # KIE SER
    "VI-LayoutXLM": {
        "rename": [],
        "transpose": [],
        "filter": [],
        "prefix_all": ["backbone.model."],
    },
    # KIE SER_RE
    "VI-LayoutXLM-RE": {
        "rename": [],
        "transpose": [
            re.compile(r"encoder.layer\.(\d+)\..*\.weight"),
            re.compile(r"visual_proj.weight"),
            re.compile(r"extractor\.ffnn_(head|tail)\.(\d+)\.weight"),
            re.compile(r"extractor\.rel_classifier.linear\.weight"),
        ],
        "filter": [],
        "prefix_all": ["backbone.model."],
    },
    # UIE - ErnieMForQuestionAnswering
    "ErnieMForQuestionAnswering": {
        "rename": [],
        "transpose": [
            re.compile(r"encoder.layer(s)*\.(\d+)\..*weight"),
            re.compile(r"qa_outputs.weight"),
        ],
        "filter": [
            re.compile(r"ernie_m.pooler\..*"),
        ],
        "prefix_all": [],
    },
    # LM - Ernie
    "Ernie": {
        "rename": [
            re.compile(r"encoder.layer(s)*\.(\d+)\..*"),
        ],
        "transpose": [
            re.compile(r"encoder.layer(s)*\.(\d+)\..*\.weight"),
            re.compile(r"classifier\.weight"),
        ],
        "filter": [],
        "prefix_all": [],
    },
    # Detection RTDETR
    "rtdetr": {
        "rename": [
        ],
        "transpose": [
            re.compile(r"transformer.query_pos_head.layers\.(\d+)\.weight"),
            re.compile(r"transformer.decoder.layers\.(\d+)\..*\.weight"),
            re.compile(r"transformer.enc_bbox_head.layers\.(\d+)\.weight"),
            re.compile(r"transformer.enc_score_head.weight"),
            re.compile(r"transformer.enc_output.0.weight"),
            re.compile(r"transformer.dec_bbox_head\.(\d+)\.layers\.(\d+)\.weight"),
            re.compile(r"transformer.dec_score_head\.(\d+)\.weight"),
            re.compile(r"neck.encoder\.(\d+)\.layers.*\.(linear.*|self_attn.out_proj)\.weight"),
        ],
        "filter": [],
        "prefix_all": [],
        "prefix": [],
        "prefix_to_save": [],
    }
}
