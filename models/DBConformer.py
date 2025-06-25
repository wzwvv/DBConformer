'''
=================================================
coding:utf-8
@Time:      2025/4/13 17:06
@File:      DBConformer.py
@Author:    Ziwei Wang
@Function:  Original model code of "DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding"
=================================================
'''

import math
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
from einops import rearrange
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        out = sum(x)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    def __init__(self, data_name, in_planes, out_planes = 64, kernel_size=63, patch_size=125, radix=2):
        nn.Module.__init__(self)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix
        self.data_name = data_name

        self.sconv = Conv(nn.Conv1d(self.in_planes, self.mid_planes, 1, bias=False, groups = radix),
                          bn=nn.BatchNorm1d(self.mid_planes), activation=None)

        self.tconv = nn.ModuleList()
        for _ in range(self.radix):
            self.tconv.append(Conv(nn.Conv1d(self.out_planes, self.out_planes, kernel_size, 1, groups=self.out_planes, padding=kernel_size // 2, bias=False,),
                                   bn=nn.BatchNorm1d(self.out_planes), activation=None))
            kernel_size //= 2

        self.interFre = InterFre()

        self.downSampling = nn.AvgPool1d(patch_size, patch_size)
        self.dp = nn.Dropout(0.5)  #

    def forward(self, x):
        N, C, T = x.shape
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [m(x) for x, m in zip(out, self.tconv)]
        out = self.interFre(out)
        if self.data_name != 'MI1-7':  # MI1-7 is the Blankertz2007 dataset in our paper
            out = out[:, :, :-1]  # Example: 14001 has 1001 time points, we exclude the final point
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class PatchEmbeddingTemporal(nn.Module):
    def __init__(self, data_name, in_planes, out_planes, kernel_size, radix, patch_size, time_points, num_classes):
        '''
        Outputs patch embeddings of shape (B, P, D)
        '''
        super().__init__()
        self.data_name = data_name
        self.stem = Stem(
            data_name=self.data_name,
            in_planes=in_planes * radix,
            out_planes=out_planes,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix
        )
        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, C, T)
        out = self.stem(x)         # (B, D, P)
        out = out.permute(0, 2, 1) # → (B, P, D)
        return out


class CrossAttention(nn.Module):
    def __init__(self, emb_size, num_heads=4, dropout=0.2):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.query_proj = nn.Linear(emb_size, emb_size)
        self.key_proj = nn.Linear(emb_size, emb_size)
        self.value_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):  # query: (B, P1, D), key_value: (B, P2, D)
        Q = self.query_proj(query)
        K = self.key_proj(key_value)
        V = self.value_proj(key_value)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        attn_scores = torch.einsum('bhqd, bhkd -> bhqk', Q, K) / (self.emb_size ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.einsum('bhqk, bhvd -> bhqd', attn_probs, V)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)

        return out


class SEBlock(nn.Module):
    """bad performance"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D = x.shape
        y = x.mean(dim=-1)  # (B, C)
        y = self.fc(y).unsqueeze(-1)  # (B, C, 1)
        return x * y


class PatchEmbeddingSpatial(nn.Module):
    def __init__(self, spa_dim, emb_size=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, spa_dim, kernel_size=25, stride=5, padding=12),  # Output: (B*C, spa_dim, T')
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),  # Output: (B*C, spa_dim, 1)
            nn.Flatten(),             # → (B*C, 16)
            nn.Linear(spa_dim, emb_size)   # → (B*C, emb_size)
        )

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        x = x.unsqueeze(2)         # (B, C, 1, T)
        x = x.reshape(B * C, 1, T)  # → (B*C, 1, T)
        x = self.encoder(x)        # → (B*C, emb_size)
        x = x.view(B, C, -1)       # → (B, C, emb_size)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out

class Gate_FC(nn.Sequential):
    def __init__(self, emb_size):
        super().__init__()
        self.fc = nn.Linear(emb_size * 2, emb_size)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class DBConformer(nn.Module):
    def __init__(self, args, emb_size=40, tem_depth=5, chn_depth=5, chn=22, n_classes=2):
        super().__init__()

        self.embedding = PatchEmbeddingTemporal(
            data_name=args.data_name,
            in_planes=args.chn,  # number of channels
            out_planes=emb_size,  # Default 40
            kernel_size=63,
            radix=1,
            patch_size=args.patch_size,  # needs to be divisible by the number of time points
            time_points=args.time_sample_num,  # number of time points
            num_classes=args.class_num  # number of classes
        )
        self.channel_embedding = PatchEmbeddingSpatial(spa_dim=args.spa_dim, emb_size=emb_size)  # Default 16
        self.P = args.time_sample_num // args.patch_size  # Example: 1000 // 125 = 8
        self.C = args.chn  # number of channels
        self.D = emb_size
        self.gate_flag = args.gate_flag  # Default False, due to the reduced performance
        self.posemb_flag = args.posemb_flag  # Default True
        self.branch = args.branch  # Default 'all', options=[all, temporal]
        self.chn_atten_flag = args.chn_atten_flag  # Default True

        if args.posemb_flag:
            self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
            self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        self.temporal_transformer = TransformerEncoder(tem_depth, emb_size)
        self.spatial_transformer = TransformerEncoder(chn_depth, emb_size)
        # self.cross_attn = CrossAttention(emb_size=emb_size, num_heads=4)  # cross-attention (bad performance)
        if args.gate_flag or self.branch == 'temporal' or self.branch == 'spatial':
            self.gate_fc = Gate_FC(emb_size)  # dual-branch weighting aggregation
            self.classifier = ClassificationHead(emb_size, n_classes)
        else:
            self.classifier = ClassificationHead(emb_size * 2, n_classes)

            if args.chn_atten_flag:
                self.spatial_attn_pool = nn.Sequential(
                    nn.Linear(emb_size, emb_size),  # D → D
                    nn.Tanh(),
                    nn.Linear(emb_size, 1),  # D → 1 (score per channel)
                )
    def forward(self, x):  # x: (B, 1, C, T)
        x = x.squeeze(1)  # → (B, C, T)
        x_embed = self.embedding(x)  # → (B, P, D)
        x_embed_spatial = self.channel_embedding(x)  # (B, C, D)
        if self.posemb_flag:
            x_embed = x_embed + self.pos_embedding_temporal  # temporal positional encoding
            x_embed_spatial = x_embed_spatial + self.pos_embedding_spatial  # spatial positional encoding

        # Temporal Transformer (attention over time dimension)
        x_temporal = self.temporal_transformer(x_embed)  # (B, P, D)
        # Spatial Transformer (attention over channels interpreted as tokens)
        x_spatial = self.spatial_transformer(x_embed_spatial)  # (B, C, D)

        if self.branch == 'temporal':
            x_fused = x_temporal.mean(dim=1)
            _, out = self.classifier(x_fused)  # out: (B, n_classes)
        elif self.branch == 'spatial':  # Using S-Conformer-only doesn't work
            x_fused = x_spatial.mean(dim=1)
            _, out = self.classifier(x_fused)  # out: (B, n_classes)
        elif self.branch == 'all':
            if self.gate_flag:
                # gated-fusion
                gate = torch.sigmoid(
                    self.gate_fc(torch.cat([x_temporal.mean(dim=1), x_spatial.mean(dim=1)], dim=-1)))  # shape: (B, D)
                x_fused = gate * x_spatial.mean(dim=1) + (1 - gate) * x_temporal.mean(dim=1)
            else:
                if self.chn_atten_flag:
                    # Attention Scores
                    x_t = x_temporal.mean(dim=1)
                    attn_scores = self.spatial_attn_pool(x_spatial)  # (B, C, 1)
                    attn_weights = torch.softmax(attn_scores, dim=1)  # (B, C, 1)
                    x_s = torch.sum(attn_weights * x_spatial, dim=1)  # (B, D)
                    x_fused = torch.cat([x_t, x_s], dim=-1)  # → (B, 2*D)
                else:
                    # Mean pooling
                    x_fused = torch.cat([
                        x_temporal.mean(dim=1),
                        x_spatial.mean(dim=1)
                    ], dim=-1)  # → (B, 2*D)
            _, out = self.classifier(x_fused)  # out: (B, n_classes)
        return x_fused, out