#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：financial_ner 
# @File    ：detection_cli_utils.py
# @Author  ：sl
# @Date    ：2022/11/7 17:37

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml
import re
from ppdettorch.core.workspace import get_registered_modules, dump_value

__all__ = ['ColorTTY', 'ArgsParser']


class ColorTTY(object):
    def __init__(self):
        super(ColorTTY, self).__init__()
        self.colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']

    def __getattr__(self, attr):
        if attr in self.colors:
            color = self.colors.index(attr) + 31

            def color_message(message):
                return "[{}m{}[0m".format(color, message)

            setattr(self, attr, color_message)
            return color_message

    def bold(self, message):
        return self.with_code('01', message)

    def with_code(self, code, message):
        return "[{}m{}[0m".format(code, message)


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config


def merge_args(config, args, exclude_args=['config', 'opt', 'slim_config']):
    for k, v in vars(args).items():
        if k not in exclude_args:
            config[k] = v
    return config


def print_total_cfg(config):
    modules = get_registered_modules()
    color_tty = ColorTTY()
    green = '___{}___'.format(color_tty.colors.index('green') + 31)

    styled = {}
    for key in config.keys():
        if not config[key]:  # empty schema
            continue

        if key not in modules and not hasattr(config[key], '__dict__'):
            styled[key] = config[key]
            continue
        elif key in modules:
            module = modules[key]
        else:
            type_name = type(config[key]).__name__
            if type_name in modules:
                module = modules[type_name].copy()
                module.update({
                    k: v
                    for k, v in config[key].__dict__.items()
                    if k in module.schema
                })
                key += " ({})".format(type_name)
        default = module.find_default_keys()
        missing = module.find_missing_keys()
        mismatch = module.find_mismatch_keys()
        extra = module.find_extra_keys()
        dep_missing = []
        for dep in module.inject:
            if isinstance(module[dep], str) and module[dep] != '<value>':
                if module[dep] not in modules:  # not a valid module
                    dep_missing.append(dep)
                else:
                    dep_mod = modules[module[dep]]
                    # empty dict but mandatory
                    if not dep_mod and dep_mod.mandatory():
                        dep_missing.append(dep)
        override = list(
            set(module.keys()) - set(default) - set(extra) - set(dep_missing))
        replacement = {}
        for name in set(override + default + extra + mismatch + missing):
            new_name = name
            if name in missing:
                value = "<missing>"
            else:
                value = module[name]

            if name in extra:
                value = dump_value(value) + " <extraneous>"
            elif name in mismatch:
                value = dump_value(value) + " <type mismatch>"
            elif name in dep_missing:
                value = dump_value(value) + " <module config missing>"
            elif name in override and value != '<missing>':
                mark = green
                new_name = mark + name
            replacement[new_name] = value
        styled[key] = replacement
    buffer = yaml.dump(styled, default_flow_style=False, default_style='')
    buffer = (re.sub(r"<missing>", r"[31m<missing>[0m", buffer))
    buffer = (re.sub(r"<extraneous>", r"[33m<extraneous>[0m", buffer))
    buffer = (re.sub(r"<type mismatch>", r"[31m<type mismatch>[0m", buffer))
    buffer = (re.sub(r"<module config missing>",
                     r"[31m<module config missing>[0m", buffer))
    buffer = re.sub(r"___(\d+)___(.*?):", r"[\1m\2[0m:", buffer)
    print(buffer)
