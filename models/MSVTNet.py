'''
=================================================
coding:utf-8
@Time:      2025/8/18 02:37
@File:      MSVTNet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class TSConv(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding='same', bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding='same', groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc)
        )


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        x += self.pe
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            seq_len,
            d_model,
            nhead,
            ff_ratio,
            Pt=0.5,
            num_layers=4,
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = PositionalEncoding(seq_len + 1, d_model)

        dim_ff = d_model * ff_ratio
        self.dropout = nn.Dropout(Pt)
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, Pt, batch_first=True, norm_first=True
        ), num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x):
        b = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(b, -1, -1), x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]


class ClsHead(nn.Sequential):
    def __init__(self, linear_in, cls):
        super().__init__(
            nn.Flatten(),
            nn.Linear(linear_in, cls),
            nn.LogSoftmax(dim=1)
        )


class MSVTNet(nn.Module):
    def __init__(
            self,
            args,
            F=[9, 9, 9, 9],
            C1=[15, 31, 63, 125],
            C2=15,
            D=2,
            P1=8,
            P2=7,
            Pc=0.3,
            nhead=8,
            ff_ratio=1,
            Pt=0.5,
            layers=2,
            b_preds=False,
    ) -> None:
        super().__init__()
        self.nCh = args.chn
        self.nTime = args.time_sample_num
        self.b_preds = b_preds
        assert len(F) == len(C1), 'The length of F and C1 should be equal.'

        self.mstsconv = nn.ModuleList([
            nn.Sequential(
                TSConv(args.chn, F[b], C1[b], C2, D, P1, P2, Pc),
                Rearrange('b d 1 t -> b t d')
            )
            for b in range(len(F))
        ])
        branch_linear_in = self._forward_flatten(cat=False)
        self.branch_head = nn.ModuleList([
            ClsHead(branch_linear_in[b].shape[1], args.class_num)
            for b in range(len(F))
        ])

        seq_len, d_model = self._forward_mstsconv().shape[1:3]  # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1]  # type: ignore
        self.last_head = ClsHead(linear_in, args.class_num)

    def _forward_mstsconv(self, cat=True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat=True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
        return x

    def forward(self, x):
        x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        x = torch.cat(x, dim=2)
        x = self.transformer(x)
        out = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x, out


class JointCrossEntoryLoss(nn.Module):
    def __init__(self, lamd: float = 0.6) -> None:
        super().__init__()
        self.lamd = lamd

    def forward(self, out, label):
        end_out = out[0]
        branch_out = out[1]
        end_loss = F.nll_loss(end_out, label)
        branch_loss = [F.nll_loss(out, label).unsqueeze(0) for out in branch_out]
        branch_loss = torch.cat(branch_loss)
        loss = self.lamd * end_loss + (1 - self.lamd) * torch.sum(branch_loss)
        return loss


if __name__ == '__main__':
    from torchinfo import summary

    net = MSVTNet(nCh=22, nTime=1000).cuda()
    print(net)
    summary(net, (64, 1, 22, 1000), depth=4)