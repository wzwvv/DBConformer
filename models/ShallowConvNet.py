'''
=================================================
coding:utf-8
@Time:      2023/12/5 17:08
@File:      ShallowConvNet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 3, 0, 0)),
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))

        self.layer1.eval()
        out = self.layer1(torch.zeros(1, 1, input_ch, input_time))
        out = torch.nn.functional.avg_pool2d(out, (1, 75), 15)
        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Linear(self.n_outputs, self.n_classes)

    def forward(self, x):
        # x = x.permute(0, 1, 3, 2)  #跑pretrian的时候关一下
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 75), 15)
        x = torch.log(x)
        x = torch.nn.functional.dropout(x)
        x = x.reshape(x.size()[0], -1)  # view to reshape
        # x = self.clf(x)
        return x
