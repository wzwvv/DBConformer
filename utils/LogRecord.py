# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 19:08
# @Author  : wenzhang
# @File    : LogRecord.py

import torch as tr
import os.path as osp
from datetime import datetime
from datetime import timedelta, timezone

from utils.utils import create_folder


class LogRecord:
    def __init__(self, args):
        self.args = args
        self.result_dir = args.result_dir
        try:
            self.data_env = 'gpu' if tr.cuda.get_device_name(0) != 'GeForce GTX 1660 Ti' else 'local'
        except Exception:
            self.data_env = 'local'
        self.data_name = args.data
        self.method = args.method
        self.align = args.align

    def log_init(self):
        create_folder(self.result_dir, self.args.data_env, self.args.local_dir)

        if self.data_env in ['local', 'mac']:
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(
                timezone(timedelta(hours=8), name='Asia/Shanghai')).strftime("%Y-%m-%d_%H_%M_%S")
        if self.data_env == 'gpu':
            time_str = datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%d_%H_%M_%S")
        if self.align:
            align_str = '_'
        else:
            align_str = '_noalign_'
        file_name_head = self.method + align_str
        self.args.out_file = open(osp.join(self.args.result_dir, file_name_head + time_str + '.txt'), 'w')
        self.args.out_file.write(self._print_args() + '\n')
        self.args.out_file.flush()
        return self.args

    def record(self, log_str):
        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        return self.args

    def _print_args(self):
        s = "==========================================\n"
        for arg, content in self.args.__dict__.items():
            s += "{}:{}\n".format(arg, content)
        return s
