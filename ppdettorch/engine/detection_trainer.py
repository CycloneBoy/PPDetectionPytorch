#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：financial_ner 
# @File    ：detection_trainer.py
# @Author  ：sl
# @Date    ：2022/11/3 17:57

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import time

from ppdettorch.data import get_categories, SniperCOCODataSet
from torch.utils.data import BatchSampler
from tqdm import tqdm

import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile

from ppdettorch.metrics import get_infer_results
from ppdettorch.utils.detection_checkpoint_utils import load_pretrain_weight
from ppdettorch.utils.detection_visualizer import visualize_results
from ppdettorch.utils.file_utils import FileUtils
from ppdettorch.utils.logger_utils import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn

from ppdettorch.optimizer import ModelEMA
from ppdettorch.core.workspace import create

# from ppdettorch.metrics.detection import Metric, COCOMetric, VOCMetric, WiderFaceMetric, get_infer_results, \
#     KeyPointTopDownCOCOEval, KeyPointTopDownMPIIEval
# from ppdettorch.metrics.detection import RBoxMetric, JDEDetMetric, SNIPERCOCOMetric

# from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
# from ppdet.utils.visualizer import visualize_results, save_result
# from ppdet.metrics import Metric, COCOMetric, VOCMetric, WiderFaceMetric, get_infer_results, KeyPointTopDownCOCOEval, KeyPointTopDownMPIIEval
# from ppdet.metrics import RBoxMetric, JDEDetMetric, SNIPERCOCOMetric
# from ppdet.data.source.sniper_coco import SniperCOCODataSet
# from ppdet.data.source.category import get_categories
# import ppdet.utils.stats as stats
# from ppdet.utils.fuse_utils import fuse_conv_bn
# from ppdet.utils import profiler
# from  ppdettorch.modeling.post_process import multiclass_nms
#
# from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter, SniperProposalsGenerator, WandbCallback
# from .export_utils import _dump_infer_config, _prune_input_spec
#
# from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients


__all__ = ['DetectionTrainer']

MOT_ARCH = ['DeepSORT', 'JDE', 'FairMOT', 'ByteTrack']

"""
检测的trainer

"""


class DetectionTrainer(object):
    """
    Detection trainer
    """

    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test'], \
            "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.use_gpu = True
        self.fp16 = True
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)

        # build data loader
        capital_mode = self.mode.capitalize()
        if cfg.architecture in MOT_ARCH and self.mode in ['eval', 'test']:
            self.dataset = self.cfg['{}MOTDataset'.format(
                capital_mode)] = create('{}MOTDataset'.format(capital_mode))()
        else:
            self.dataset = self.cfg['{}Dataset'.format(capital_mode)] = create(
                '{}Dataset'.format(capital_mode))()

        if cfg.architecture == 'DeepSORT' and self.mode == 'train':
            logger.error('DeepSORT has no need of training on mot dataset.')
            sys.exit(1)

        if cfg.architecture == 'FairMOT' and self.mode == 'eval':
            images = self.parse_mot_images(cfg)
            self.dataset.set_images(images)

        if self.mode == 'train':
            self.loader = create('{}Reader'.format(capital_mode))(
                self.dataset, cfg.worker_num)

        if cfg.architecture == 'JDE' and self.mode == 'train':
            cfg['JDEEmbeddingHead'][
                'num_identities'] = self.dataset.num_identities_dict[0]
            # JDE only support single class MOT now.

        if cfg.architecture == 'FairMOT' and self.mode == 'train':
            cfg['FairMOTEmbeddingHead'][
                'num_identities_dict'] = self.dataset.num_identities_dict
            # FairMOT support single class and multi-class MOT now.

        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.architecture)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        if cfg.architecture == 'YOLOX':
            for k, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3  # for amp(fp16)
                    m.momentum = 0.97  # 0.03 in pytorch

        # normalize params for deploy
        if 'slim' in cfg and cfg['slim_type'] == 'OFA':
            self.model.model.load_meanstd(cfg['TestReader'][
                                              'sample_transforms'])
        elif 'slim' in cfg and cfg['slim_type'] == 'Distill':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                                                      'sample_transforms'])
        elif 'slim' in cfg and cfg[
            'slim_type'] == 'DistillPrune' and self.mode == 'train':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                                                      'sample_transforms'])
        else:
            self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            if cfg.architecture == 'FairMOT':
                self.loader = create('EvalMOTReader')(self.dataset, 0)
            else:
                self._eval_batch_sampler = BatchSampler(
                    self.dataset, batch_size=self.cfg.EvalReader['batch_size'], drop_last=False)
                reader_name = '{}Reader'.format(self.mode.capitalize())
                # If metric is VOC, need to be set collate_batch=False.
                if cfg.metric == 'VOC':
                    cfg[reader_name]['collate_batch'] = False
                self.loader = create(reader_name)(self.dataset, cfg.worker_num,
                                                  self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

            # Unstructured pruner is only enabled in the train mode.
            if self.cfg.get('unstructured_prune'):
                self.pruner = create('UnstructuredPruner')(self.model,
                                                           steps_per_epoch)
        if self.use_amp and self.amp_level == 'O2':
            self.model, self.optimizer = torch.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.amp_level)
        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_black_list = self.cfg.get('ema_black_list', None)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list)

        self._nranks = 1
        self._local_rank = 1

        self.status = {}

        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        # # initial default callbacks
        # self._init_callbacks()
        #
        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def parse_mot_images(self, cfg):
        import glob
        # for quant
        dataset_dir = cfg['EvalMOTDataset'].dataset_dir
        data_root = cfg['EvalMOTDataset'].data_root
        data_root = '{}/{}'.format(dataset_dir, data_root)
        seqs = os.listdir(data_root)
        seqs.sort()
        all_images = []
        for seq in seqs:
            infer_dir = os.path.join(data_root, seq)
            assert infer_dir is None or os.path.isdir(infer_dir), \
                "{} is not a directory".format(infer_dir)
            images = set()
            exts = ['jpg', 'jpeg', 'png', 'bmp']
            exts += [ext.upper() for ext in exts]
            for ext in exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            images.sort()
            assert len(images) > 0, "no image found in {}".format(infer_dir)
            all_images.extend(images)
            logger.info("Found {} inference images in total.".format(
                len(images)))
        return all_images

    def load_weights(self, weights, do_transform=False, output_dir=None):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0

        load_pretrain_weight(self.model, weights, do_transform=do_transform, output_dir=output_dir)
        logger.debug("Load weights {} to start training".format(weights))

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        clsid2catid, catid2name = self.get_categories_mapping()

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        device = torch.device("cuda")
        if self.use_gpu:
            if self.fp16:
                self.model = self.model.half()
                logger.info(f"使用FP16量化模型推理")
            self.model.to(device)

        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            for k, v in data.items():
                v_t = torch.from_numpy(v).to(device)
                if self.fp16:
                    v_t = v_t.half()
                data[k] = v_t

            with torch.no_grad():
                outs = self.model(data)

            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.cpu().numpy()
            results.append(outs)

        # sniper
        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                        if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                        if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                        if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                        if 'keypoint' in batch_res else None
                    image = visualize_results(
                        image, bbox_res, mask_res, segm_res, keypoint_res,
                        int(im_id), catid2name, draw_threshold)
                    self.status['result_image'] = np.array(image.copy())
                    # if self._compose_callback:
                    #     self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_name = self._get_save_image_name(output_dir,
                                                          image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_name))
                    FileUtils.check_file_exists(output_dir)
                    image.save(save_name, quality=95)

                    start = end

        return results

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg['classwise'] if 'classwise' in self.cfg else False
        if self.cfg.metric == 'COCO' or self.cfg.metric == "SNIPERCOCO":
            # TODO: bias should be unified
            bias = 1 if self.cfg.get('bias', False) else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} \
                if self.mode == 'eval' else None

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset
            else:
                dataset = self.dataset
                anno_file = dataset.get_anno()

            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            if self.cfg.metric == "COCO":
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
            elif self.cfg.metric == "SNIPERCOCO":  # sniper
                self._metrics = [
                    SNIPERCOCOMetric(
                        anno_file=anno_file,
                        dataset=dataset,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
        elif self.cfg.metric == 'RBOX':
            # TODO: bias should be unified
            bias = self.cfg['bias'] if 'bias' in self.cfg else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            imid2path = self.cfg.get('imid2path', None)

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            anno_file = self.dataset.get_anno()
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()

            self._metrics = [
                RBoxMetric(
                    anno_file=anno_file,
                    classwise=classwise,
                    output_eval=output_eval,
                    bias=bias,
                    save_prediction_only=save_prediction_only,
                    imid2path=imid2path)
            ]
        elif self.cfg.metric == 'VOC':
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            self._metrics = [
                VOCMetric(
                    label_list=self.dataset.get_label_list(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type,
                    classwise=classwise,
                    output_eval=output_eval,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'WiderFace':
            multi_scale = self.cfg.multi_scale_eval if 'multi_scale_eval' in self.cfg else True
            self._metrics = [
                WiderFaceMetric(
                    image_dir=os.path.join(self.dataset.dataset_dir,
                                           self.dataset.image_dir),
                    anno_file=self.dataset.get_anno(),
                    multi_scale=multi_scale)
            ]
        elif self.cfg.metric == 'KeyPointTopDownCOCOEval':
            eval_dataset = self.cfg['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg.num_joints,
                    self.cfg.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'KeyPointTopDownMPIIEval':
            eval_dataset = self.cfg['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownMPIIEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg.num_joints,
                    self.cfg.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'MOTDet':
            self._metrics = [JDEDetMetric(), ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext

    def predict_simple(self,
                       images,
                       draw_threshold=0.5,
                       output_dir='output',
                       save_results=False,
                       visualize=True):
        """
        简单推理

        :param images:
        :param draw_threshold:
        :param output_dir:
        :param save_results:
        :param visualize:
        :return:
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        device = torch.device("cuda")
        if self.use_gpu:
            if self.fp16:
                self.model = self.model.half()
                logger.info(f"使用FP16量化模型推理")
            self.model.to(device)

        # data = {
        #     "image": images
        # }
        data = images
        for k, v in data.items():
            v_t = torch.from_numpy(v).to(device)
            if self.fp16:
                v_t = v_t.half()
            data[k] = v_t

        with torch.no_grad():
            self.model.eval()
            outs = self.model(data)

        # for key in ['im_shape', 'scale_factor', 'im_id']:
        #     if isinstance(data, typing.Sequence):
        #         outs[key] = data[0][key]
        #     else:
        #         outs[key] = data[key]
        for key, value in outs.items():
            if hasattr(value, 'numpy'):
                outs[key] = value.cpu().numpy()

        return outs

    def get_categories_mapping(self):
        """
        获取 预测的类别信息

        :return:
        """
        if self.cfg.predict_labels is not None:
            anno_file = self.cfg.predict_labels
        else:
            anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)
        return clsid2catid, catid2name
