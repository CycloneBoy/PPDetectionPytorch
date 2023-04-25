#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2023/4/25 - 21:35
import os
import pickle
import sys

import numpy as np
import six

from .base_utils import BaseUtil

'''
文件处理的工具类

'''


class _open_buffer(object):

    def __init__(self, buffer):
        self.buffer = buffer

    def __enter__(self):
        return self.buffer


class _buffer_reader(_open_buffer):

    def __init__(self, buffer):
        super(_buffer_reader, self).__init__(buffer)
        self.initial_tell = self.buffer.tell()

    def __exit__(self, *args):
        # `args[0]` is type of exception. When the `read` is abnormal, the file pointer returns to the initial position.
        if args[0] is not None:
            self.buffer.seek(self.initial_tell)


class _buffer_writer(_open_buffer):

    def __exit__(self, *args):
        self.buffer.flush()


def _is_file_path(path):
    return isinstance(path, str)


def _open_file_buffer(path_or_buffer, mode):
    if _is_file_path(path_or_buffer):
        return open(path_or_buffer, mode)
    else:
        if 'w' in mode:
            return _buffer_writer(path_or_buffer)
        elif 'r' in mode:
            return _buffer_reader(path_or_buffer)
        else:
            raise ValueError(
                "Expected 'r' or 'w' in mode but got {}".format(mode))


def _pickle_loads_mac(path, f):
    pickle_bytes = bytearray(0)
    file_size = os.path.getsize(path)
    max_bytes = 2 ** 30
    for _ in range(0, file_size, max_bytes):
        pickle_bytes += f.read(max_bytes)
    load_result = pickle.loads(pickle_bytes, encoding='latin1')
    return load_result


def _pack_loaded_dict(load_obj):
    if isinstance(load_obj, dict):
        unpack_info = 'UnpackBigParamInfor@@'
        if unpack_info in load_obj:
            removes = []
            for key, value in load_obj[unpack_info].items():
                slices = [load_obj[part] for part in value["slices"]]
                load_obj[key] = np.concatenate(slices).reshape(
                    value["OriginShape"])
                removes += value["slices"]
            for key in removes:
                load_obj.pop(key)
            load_obj.pop(unpack_info)

    return load_obj


def _ndarray_to_tensor(obj, return_numpy):
    if return_numpy:
        return obj
    else:
        import torch
        return torch.tensor(obj)


def _parse_load_config(configs):
    supported_configs = [
        'model_filename',
        'params_filename',
        'keep_name_table',
        'return_numpy',
    ]

    # input check
    for key in configs:
        if key not in supported_configs:
            raise ValueError(
                "The additional config (%s) of `paddle.load` is not supported."
                % key
            )

    # construct inner config
    inner_config = _SaveLoadConfig()
    inner_config.model_filename = configs.get('model_filename', None)
    inner_config.params_filename = configs.get('params_filename', None)
    inner_config.keep_name_table = configs.get('keep_name_table', None)
    inner_config.return_numpy = configs.get('return_numpy', False)

    return inner_config


class _SaveLoadConfig(object):

    def __init__(self):
        self._output_spec = None
        self._model_filename = None
        self._params_filename = None
        self._separate_params = False
        # used for `paddle.load`
        self._keep_name_table = False

        # NOTE: Users rarely use following configs, so these configs are not open to users,
        # reducing user learning costs, but we retain the configuration capabilities

        # If True, programs are modified to only support direct inference deployment.
        # Otherwise,more information will be stored for flexible optimization and re-training.
        # Currently, only True is supported
        self._export_for_deployment = True

        # If True, It will save inference program only, and do not save params of Program
        self._program_only = False
        self.with_hook = False

        # if True, multi `StaticFunction` will share params in one file.
        self.combine_params = False

    @property
    def output_spec(self):
        return self._output_spec

    @output_spec.setter
    def output_spec(self, spec):
        if spec is None:
            return
        if not isinstance(spec, list):
            raise TypeError(
                "The config `output_spec` should be 'list', but received input type is %s."
                % type(input))
            for var in spec:
                if not isinstance(var, core.VarBase):
                    raise TypeError(
                        "The element in config `output_spec` list should be 'Variable', but received element's type is %s."
                        % type(var))
        self._output_spec = spec

    @property
    def model_filename(self):
        return self._model_filename

    @model_filename.setter
    def model_filename(self, filename):
        if filename is None:
            return
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The config `model_filename` should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError("The config `model_filename` is empty string.")
        self._model_filename = filename

    @property
    def params_filename(self):
        return self._params_filename

    @params_filename.setter
    def params_filename(self, filename):
        if filename is None:
            return
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The config `params_filename` should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError("The config `params_filename` is empty string.")
        self._params_filename = filename

    @property
    def keep_name_table(self):
        return self._keep_name_table

    @keep_name_table.setter
    def keep_name_table(self, value):
        if value is None:
            return
        if not isinstance(value, bool):
            raise TypeError(
                "The config `keep_name_table` should be bool value, but received input's type is %s."
                % type(value))
        self._keep_name_table = value


def load(path, **configs):
    '''
    Load an object can be used in paddle from specified path.

    Note:
        Now supports loading ``state_dict`` of Layer/Optimizer, Tensor and nested structure containing Tensor, Program.

    Note:
        In order to use the model parameters saved by paddle more efficiently,
        ``paddle.load`` supports loading ``state_dict`` of Layer from the result of
        other save APIs except ``paddle.save`` , but the argument ``path`` format is
        different:
        1. loading from ``paddle.static.save`` or ``paddle.Model().save(training=True)`` ,
        ``path`` needs to be a complete file name, such as ``model.pdparams`` or
        ``model.pdopt`` ;
        2. loading from ``paddle.jit.save`` or ``paddle.static.save_inference_model``
        or ``paddle.Model().save(training=False)`` , ``path`` need to be a file prefix,
        such as ``model/mnist``, and ``paddle.load`` will get information from
        ``mnist.pdmodel`` and ``mnist.pdiparams`` ;
        3. loading from paddle 1.x APIs ``paddle.fluid.io.save_inference_model`` or
        ``paddle.fluid.io.save_params/save_persistables`` , ``path`` need to be a
        directory, such as ``model`` and model is a directory.

    Note:
        If you load ``state_dict`` from the saved result of static mode API such as
        ``paddle.static.save`` or ``paddle.static.save_inference_model`` ,
        the structured variable name in dynamic mode will cannot be restored.
        You need to set the argument ``use_structured_name=False`` when using
        ``Layer.set_state_dict`` later.

    Args:
        path(str|BytesIO) : The path/buffer to load the target object. Generally, the path is the target
            file path. When loading state_dict from the saved result of the API used to save
            the inference model, the path may be a file prefix or directory.
        **configs (dict, optional): other load configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x
            ``save_inference_model`` save format. Default file name is :code:`__model__` .
            (2) params_filename (str): The persistable variables file name of the paddle 1.x
            ``save_inference_model`` save format. No default file name, save variables separately
            by default.
            (3) return_numpy(bool): If specified as True, return tensor as numpy.ndarray, otherwise return tensor as paddle.Tensor.
            Default False.

    Returns:
        Object(Object): a target object can be used in paddle

    Examples:
        .. code-block:: python

            # example 1: dynamic graph
            import paddle
            emb = paddle.nn.Embedding(10, 10)
            layer_state_dict = emb.state_dict()

            # save state_dict of emb
            paddle.save(layer_state_dict, "emb.pdparams")

            scheduler = paddle.optimizer.lr.NoamDecay(
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            opt_state_dict = adam.state_dict()

            # save state_dict of optimizer
            paddle.save(opt_state_dict, "adam.pdopt")
            # save weight of emb
            paddle.save(emb.weight, "emb.weight.pdtensor")

            # load state_dict of emb
            load_layer_state_dict = paddle.load("emb.pdparams")
            # load state_dict of optimizer
            load_opt_state_dict = paddle.load("adam.pdopt")
            # load weight of emb
            load_weight = paddle.load("emb.weight.pdtensor")


            # example 2: Load multiple state_dict at the same time
            from paddle import nn
            from paddle.optimizer import Adam

            layer = paddle.nn.Linear(3, 4)
            adam = Adam(learning_rate=0.001, parameters=layer.parameters())
            obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
            path = 'example/model.pdparams'
            paddle.save(obj, path)
            obj_load = paddle.load(path)


            # example 3: static graph
            import paddle
            import paddle.static as static

            paddle.enable_static()

            # create network
            x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
            z = paddle.static.nn.fc(x, 10)

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            for var in prog.list_vars():
                if list(var.shape) == [224, 10]:
                    tensor = var.get_value()
                    break

            # save/load tensor
            path_tensor = 'temp/tensor.pdtensor'
            paddle.save(tensor, path_tensor)
            load_tensor = paddle.load(path_tensor)

            # save/load state_dict
            path_state_dict = 'temp/model.pdparams'
            paddle.save(prog.state_dict("param"), path_tensor)
            load_state_dict = paddle.load(path_tensor)


            # example 4: load program
            import paddle

            paddle.enable_static()

            data = paddle.static.data(
                name='x_static_save', shape=(None, 224), dtype='float32')
            y_static = z = paddle.static.nn.fc(data, 10)
            main_program = paddle.static.default_main_program()
            path = "example/main_program.pdmodel"
            paddle.save(main_program, path)
            load_main = paddle.load(path)
            print(load_main)


            # example 5: save object to memory
            from io import BytesIO
            import paddle
            from paddle.nn import Linear
            paddle.disable_static()

            linear = Linear(5, 10)
            state_dict = linear.state_dict()
            byio = BytesIO()
            paddle.save(state_dict, byio)
            tensor = paddle.randn([2, 3], dtype='float32')
            paddle.save(tensor, byio)
            byio.seek(0)
            # load state_dict
            dict_load = paddle.load(byio)

    '''

    if os.path.isfile(path):
        config = _parse_load_config(configs)
        exception_type = pickle.UnpicklingError
        try:
            with _open_file_buffer(path, 'rb') as f:
                # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
                if (
                        _is_file_path(path)
                        and sys.platform == 'darwin'
                        and sys.version_info.major == 3
                ):
                    load_result = _pickle_loads_mac(path, f)
                else:
                    load_result = pickle.load(f, encoding='latin1')

                # TODO(weixin):If `obj` is any object, the judgment condition should be more precise.
                if isinstance(load_result, dict):
                    load_result = _pack_loaded_dict(load_result)
                    # paddle2.0: paddle.save/load
                    if "StructuredToParameterName@@" in load_result:

                        for key in load_result["StructuredToParameterName@@"]:
                            if isinstance(load_result[key], np.ndarray):
                                load_result[key] = _ndarray_to_tensor(
                                    load_result[key], config.return_numpy
                                )

                        if (
                                not config.keep_name_table
                                and "StructuredToParameterName@@" in load_result
                        ):
                            del load_result["StructuredToParameterName@@"]
                    else:
                        pass
                        # paddle2.1 static.save/load
                        # load_result = _parse_load_result(
                        #     load_result, config.return_numpy
                        # )

                else:
                    pass
                    # load_result = _parse_load_result(
                    #     load_result, config.return_numpy
                    # )

        except exception_type as msg_pickle:
            raise ValueError(
                "`paddle.load` can not parse the file:{}.".format(
                    path
                )
            )

    else:
        raise ValueError(
            "`paddle.load` can not parse the file:{}.".format(
                path
            )
        )

    return load_result


class ModelUtils(BaseUtil):
    """
    文件工具类
    """

    def init(self):
        pass

    @staticmethod
    def load(path, **configs):
        """
        paddle.load

        :param path:
        :param configs:
        :return:
        """
        return load(path, **configs)
