'''
=================================================
coding:utf-8
@Time:      2024/9/28 21:23
@File:      FBMSNet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchsummary import summary

import sys

current_module = sys.modules[__name__]

debug = False

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

# %% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''

    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''

    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

#%% support of mixConv2d
import torch.nn.functional as F

from typing import Tuple, Optional

def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def _get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    NOTE: This does not currently work with torch.jit.script
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSameExport, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = (0, 0)

    def forward(self, x):
        input_size = x.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size

        if self.pad is not None:
            x = self.pad(x)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
            else:
                # dynamic padding
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
    return padding, dynamic

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if isinstance(kernel_size, tuple):
            padding = (0,padding)
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        # import  numpy as  np
        # equal_ch = True
        # groups = len(kernel_size)
        # if equal_ch:  # 均等划分通道
        #     in_splits = _split_channels(in_channels, num_groups)
        #     out_splits = _split_channels(out_channels, num_groups)
        # else:  # 指数划分通道
        #     in_splits = _split_channels(in_channels, num_groups)


        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x

#%% FBMSNet_MixConv
class FBMSNet(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, args, nChan, nTime, temporalLayer='LogVarLayer', dilatability=8, dropoutP=0.5, **kwargs):
        # input_size: channel x datapoint
        super(FBMSNet, self).__init__()
        nClass = args.class_num
        num_Feat = args.feature_deep_dim
        in_channels = args.in_chans
        self.strideFactor = 3
        self.mixConv2d = nn.Sequential(
            MixedConv2d(in_channels,
                        out_channels=num_Feat,
                        kernel_size=[(1,15),(1,31),(1,63),(1,125)],
                        stride=1, padding='', dilation=1, depthwise=False,),
            nn.BatchNorm2d(num_Feat),
        )
        self.scb = self.SCB(in_chan=num_Feat, out_chan=num_Feat*dilatability, nChan=int(nChan))

        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        # size = self.get_size(args, nChan, nTime)

        # self.fc = self.LastBlock(size[1], nClass)

    def forward(self, x):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        print(x.shape)
        y = self.mixConv2d(x)
        x = self.scb(y)
        print(x.shape)
        input('')
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        f = torch.flatten(x, start_dim=1)
        # c = self.fc(f)
        return f

    def get_size(self, args, nChan, nTime):
        data = torch.ones((1, args.in_chans, nChan, nTime))
        x = self.mixConv2d(data)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()

#%% FBMSNet_Inception
class FBMSNet_Inception(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, nChan, nTime, nClass=4,sampling_rate=250, temporalLayer='LogVarLayer', num_Feat=36, dilatability=8, dropoutP=0.5, *args, **kwargs):
        # input_size: channel x datapoint
        super(FBMSNet_Inception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625]
        self.strideFactor = 4

        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=(0,int(self.inception_window[0] * sampling_rate/2))),
            nn.BatchNorm2d(9),
        )
        self.Tception2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[1] * sampling_rate-1)), stride=1, padding=(0,int(self.inception_window[1] * sampling_rate/2-1))),
            nn.BatchNorm2d(9),
        )
        self.Tception3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=(0,int(self.inception_window[2] * sampling_rate/2))),
            nn.BatchNorm2d(9),
        )
        self.Tception4 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[3] * sampling_rate)), stride=1, padding=(0,int(self.inception_window[3] * sampling_rate/2))),
            nn.BatchNorm2d(9),
        )
        self.scb = self.SCB(in_chan=36, out_chan=288, nChan=int(nChan))

        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        size = self.get_size(nChan, nTime)

        self.fc = self.LastBlock(size[1],nClass)

    def forward(self, x):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=1)
        y = self.Tception4(x)
        out = torch.cat((out, y), dim=1)
        out = self.scb(out)
        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[3] / self.strideFactor)])
        out = self.temporalLayer(out)
        f = torch.flatten(out, start_dim=1)
        c = self.fc(f)
        return c,f

    def get_size(self, nChan, nTime):
        data = torch.ones((1, 9, nChan, nTime))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=1)
        y = self.Tception4(data)
        out = torch.cat((out, y), dim=1)
        x = self.scb(out)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()

#%% __main__
if __name__ == "__main__":

    net = FBMSNet_Inception(nChan=22, nTime=512).cuda()
    summary(net, (9, 22, 512))

    net = FBMSNet(nChan=22, nTime=512).cuda()
    summary(net, (9, 22, 512))