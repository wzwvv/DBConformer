'''
=================================================
coding:utf-8
@Time:      2024/1/29 16:48
@File:      EEGNeX.py
@Author:    Ziwei Wang
@Function: EEGNeX reproduce
=================================================
'''
import torch.nn as nn
class EEGNeX(nn.Module):
    def __init__(self, classes_num, in_channels=22, time_step=250*3):
        super(EEGNeX, self).__init__()
        self.drop_out = 0.5
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False,
                padding=(1 // 2, 64 // 2)
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8),  # output shape (8, C, T)
            nn.ELU(),
            nn.Conv2d(
                in_channels=8,  # input shape (1, C, T)
                out_channels=32,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False,
                padding=(1 // 2, 64 // 2)
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(32)  # output shape (8, C, T)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # input shape (8, C, T)
                out_channels=64,  # num_filters
                kernel_size=(in_channels, 1),  # filter size, 一般取(C, 1)
                groups=32,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(64),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # input shape (16, 1, T//4)
                out_channels=32,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=32,
                bias=False,
                padding=(1 // 2, 16 // 2),
                dilation=(1, 2)
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(32),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=32,  # input shape (16, 1, T//4)
                out_channels=8,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=8,
                bias=False,
                padding=(1 // 2, 16 // 2),
                dilation=(1, 4)
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(8),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )
        self.out = nn.Linear((8 * (time_step // 32) // 3), classes_num)

    def forward(self, x):
        x = x.unsqueeze(1)  # [N, C, T] -> [N, 1, C, T]
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        # x = self.out(x)
        return x