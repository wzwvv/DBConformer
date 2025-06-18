'''
=================================================
coding:utf-8
@Time:      2024/9/27 21:04
@File:      ADFCNN.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import torch
import torch.nn.functional as F
from torch import nn
import math


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LazyLinearWithConstraint(nn.LazyLinear):
    def __init__(self, *args, max_norm=1., **kwargs):
        super(LazyLinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return self(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=16, dim=16, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos

class ADFCNN(nn.Module):
    def __init__(self,
                 num_channels: int,
                 sampling_rate: int,
                 F1=8, D=1, F2='auto', P1=4, P2=8, pool_mode='mean',
                 drop_out=0.25, layer_scale_init_value=1e-6, nums=4):
        super(ADFCNN, self).__init__()

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        pooling_size = 0.3
        hop_size = 0.7
        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 125], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 30], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )

        # Spatial
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, (num_channels, 1), padding=0, groups=F2, bias=False, max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, 32), stride=32),
            nn.Dropout(drop_out),
        )

        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, kernel_size=[num_channels, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, 75), stride=25),
            ActLog(),
            nn.Dropout(drop_out),

        )

        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(drop_out)
        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)

        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)  #### the last one is channel

        B, N, C = x_attention.shape

        q = self.w_q(x_attention).permute(0, 2, 1)
        k = self.w_k(x_attention).permute(0, 2, 1)
        v = self.w_v(x_attention).permute(0, 2, 1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        d_k = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        # -------------------
        attn = attn.softmax(dim=-1)

        x = (attn @ v).reshape(B, N, C)

        x_attention = x_attention + self.drop(x)
        x_attention = x_attention.reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)
        x = self.drop(x_attention)
        x = x.reshape(x.shape[0], -1)
        return x


class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(8, num_classes, (1, 51)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.dense(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x


class ADFCNet(nn.Module):
    def __init__(self,
                 num_classes: 2,
                 num_channels: int,
                 sampling_rate: int):
        super(ADFCNet, self).__init__()

        self.backbone = ADFCNN(num_channels=num_channels, sampling_rate=sampling_rate)
        self.classifier = classifier(num_classes)

    def forward(self, x):
        fea = self.backbone(x)
        pred = self.classifier(fea)
        return fea, pred


def get_model(args):
    model = ADFCNet(num_classes=args.class_num,
                num_channels=args.chn,
                sampling_rate=args.sample_rate)

    return model


class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))