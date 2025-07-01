'''
=================================================
coding:utf-8
@Time:      2025/6/24 20:21
@File:      SlimSeiz.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import os
import sys
import random
import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat, einsum

# device = "cpu"

# https://blog.csdn.net/cskywit/article/details/137448871
# https://github.com/johnma2006/mamba-minimal/blob/master/model.py

class MambaBlock(nn.Module):
    def __init__(self, input_channels):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()

        self.d_model = input_channels
        self.d_inner = self.d_model * 2
        self.dt_rank = math.ceil(self.d_model / 16)
        self.d_state = 16

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)  # , bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            # bias=args.conv_bias,
            kernel_size=3,
            groups=self.d_inner,
            padding=2,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)  # , bias=args.bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


# 1 dimentional version + Mamba
## Patient-Specific Seizure Prediction via Adder Network and Supervised Contrastive Learning
## ADDNet-SCL-1d

class SlimSeiz(nn.Module):
    def __init__(self, input_channels=3):
        super(SlimSeiz, self).__init__()
        self.input_channels = input_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=21, stride=1, padding=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv4_1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=2),
            nn.ReLU()
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # self.conv5_1 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2),
        #     nn.ReLU()
        # )
        # self.conv5_2 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )

        self.mixer = MambaBlock(32)
        self.norm = RMSNorm(32)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        # print(x.shape)
        x = x.squeeze(1)  # → (B, C, T)
        x = self.conv1(x)
        x = self.pool3(self.conv2_1(x) + self.conv2_2(x))
        x = self.conv4_1(x) + self.conv4_2(x)
        # x = self.conv5_1(x)+self.conv5_2(x)
        # x = [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        x = self.mixer(self.norm(x)) + x
        x = x.permute(0, 2, 1)  # [batch, channels, seq_len]
        x = self.adaptive_avg_pool(x)
        x_digits = x.contiguous().view(x.size(0), -1)
        # print(f'x.shape = {x.shape}')
        x_res = self.classifier(x_digits)
        return x_digits, x_res