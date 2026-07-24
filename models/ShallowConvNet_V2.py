'''
=================================================
coding:utf-8
@Time:      2025/5/24 12:00
@File:      ShallowConvNet_V2.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import torch
import torch.nn as nn
from typing import Optional


class ShallowConvNetV2(nn.Module):
    def __init__(
        self,
        Chans: int,
        Samples: int,
        dropoutRate: Optional[float] = 0.5, midDim: Optional[int] = 40,
    ):
        super(ShallowConvNetV2, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=midDim, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=midDim,
                      out_channels=midDim,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=midDim),
            nn.ELU(), #Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(), # Activation('log'),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        return output

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)