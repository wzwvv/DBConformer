# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : network.py
import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.EEGNet import EEGNet_feature, EEGNet
from models.FC import FC, FC_xy, FC_wave, FC_wave_xy, FC_xy_2l
from models.DeepConvNet import DeepConvNet
from models.ShallowConvNet import ShallowConvNet
from models.ShallowConvNet_V2 import ShallowConvNetV2
from models.EEGNeX import EEGNeX
from models.IFNetV2 import IFNet  # IFNetV2
from models.FBCNet import FBCNet
from models.ADFCNN import ADFCNN
from models.Conformer import Conformer
from models.FBMSNet import FBMSNet
from models.DBConformer import DBConformer
from models.DBConformer_V2 import DBConformerV2
from models.CTNet import CTNet


from models.IFMambaNet import IFMambaNet



def backbone_net(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = EEGNet_feature(n_classes=args.class_num,
                          Chans=args.chn,
                          Samples=args.time_sample_num,
                          kernLenght=int(args.sample_rate // 2),
                          F1=F1,
                          D=D,
                          F2=F2,
                          dropoutRate=args.dropoutRate,  # TODO: 0.25 in within, 0.5 in cross-subject
                          norm_rate=0.5)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_ifnet(args, return_type='xy'):
    netF = IFNet(args.data_name, args.chn, args.embed_dims, kernel_size=63, radix=1, patch_size=args.patch_size, time_points=args.time_sample_num, num_classes=args.class_num)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_scnn_v2(args, return_type='xy'):
    netF = ShallowConvNetV2(Chans=args.chn, Samples=args.time_sample_num, dropoutRate=args.dropout_rate, midDim=40)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_CTNet(args, return_type='xy'):
    netF = CTNet(
        heads=2,
        emb_size=16,
        depth=6,
        eeg1_f1=8,
        eeg1_D=2,
        eeg1_kernel_size=64,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=args.dropout_rate,
        eeg1_number_channel=args.chn,
        number_class=args.class_num,
        flatten_eeg1=240)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC

def backbone_net_ifmambanet(args, return_type='xy'):
    netF = IFMambaNet(in_channels=args.chn, num_classes=args.class_num, seq_len=args.time_sample_num, hidden_dim=args.embed_dims)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC

def backbone_net_fbmsnet(args, return_type='xy'):
    netF = FBMSNet(args, nChan=args.chn,  nTime=args.time_sample_num)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC

def backbone_net_adfcnn(args, return_type='xy'):
    netF = ADFCNN(num_channels=args.chn, sampling_rate=args.sample_rate)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC

def backbone_net_fbcnet(args, return_type='xy'):
    netF = FBCNet(data_name=args.data_name, nChan=args.chn, nTime=args.time_sample_num, nClass=args.class_num, nBands=args.nBands)
    return netF

def backbone_net_conformer(args, return_type='xy'):
    netF = Conformer(args, emb_size=40, depth=6, chn=args.chn, n_classes=args.class_num)
    return netF


def backbone_net_dbconformer(args):
    netF = DBConformer(args, emb_size=args.emb_size, tem_depth=args.transformer_depth_tem, chn_depth=args.transformer_depth_chn, chn=args.chn, n_classes=args.class_num)  # TODO
    return netF

def backbone_net_dbconformer_plot(args):
    netF = DBConformerV2(args, emb_size=args.emb_size, tem_depth=args.transformer_depth_tem, chn_depth=args.transformer_depth_chn, chn=args.chn, n_classes=args.class_num)  # TODO
    return netF


def backbone_net_eegnex(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = EEGNeX(classes_num=args.class_num,
                  in_channels=args.chn,
                  time_step=args.time_sample_num)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_wave(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = EEGWaveNet(n_chans=args.chn, n_classes=args.class_num)
    if return_type == 'y':
        netC = FC_wave(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_wave_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_deep(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = DeepConvNet(n_classes=args.class_num,
                       input_ch=args.chn,
                       input_time=args.time_sample_num,
                       batch_norm=True,
                       batch_norm_alpha=0.1)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_shallow(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = ShallowConvNet(n_classes=args.class_num,
                          input_ch=args.chn,
                          input_time=args.time_sample_num,
                          batch_norm=True,
                          batch_norm_alpha=0.1)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC



def netClf(args, return_type):
    if return_type == 'y':
        netC = FC(args.projector_dim2, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.projector_dim2, args.class_num)
    return netC



def netFea(args):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = EEGNet_feature(n_classes=args.class_num,
                          Chans=args.chn,
                          Samples=args.time_sample_num,
                          kernLenght=int(args.sample_rate // 2),
                          F1=F1,
                          D=D,
                          F2=F2,
                          dropoutRate=0.25,
                          norm_rate=0.5)
    return netF


# def encoder(args, nhead, nlayer):
#     encoder_layers = TransformerEncoderLayer(args.dim_e, dim_feedforward=2 * args.dim_e,
#                                              nhead=nhead, )  # 2 heads in EEGNet, 5 in shallow
#     transformer_encoder = TransformerEncoder(encoder_layers, nlayer).cuda()  # 2 layers in EEGNet, 4 in shallow
#     return transformer_encoder
#
#
# def projector(args):
#     projector_t = nn.Sequential(
#         nn.Linear(args.dim_p, args.projector_dim1),  # TODO multiply configs.input_channels
#         nn.BatchNorm1d(args.projector_dim1),
#         nn.ReLU(),  # delete nonlinear --> worse performance
#         nn.Linear(args.projector_dim1, args.projector_dim2),
#         nn.BatchNorm1d(args.projector_dim2),
#         nn.ReLU(),  # delete nonlinear --> worse performance
#     ).cuda()
#     return projector_t

def encoder(args, nhead, nlayer):
    """Transformer Encoder处理脑电信号"""
    encoder_layer = TransformerEncoderLayer(
        d_model=args.dim_e,  # 维度
        nhead=nhead,  # 多头注意力
        dim_feedforward=max(2 * args.dim_e, 128),  # 动态设置dim_feedforward
        dropout=0.1,  # 避免过拟合
        batch_first=True  # 使得 (N, T, C) 格式兼容
    )
    transformer_encoder = TransformerEncoder(encoder_layer, num_layers=nlayer)
    return transformer_encoder

def projector(args):
    """用于降维和特征提取"""
    projector_t = nn.Sequential(
        nn.Linear(args.dim_p, args.projector_dim1),
        nn.LayerNorm(args.projector_dim1),  # within场景替换 BatchNorm1d 更适合小批量 EEG 训练
        nn.ReLU(inplace=True),
        nn.Linear(args.projector_dim1, args.projector_dim2),
        nn.Tanh()  # 归一化输出
    )
    return projector_t


def encoder_t(args):
    encoder_layers = TransformerEncoderLayer(args.dim_e, dim_feedforward=2 * args.dim_e,
                                             nhead=2, )  # 2 heads in EEGNet, 5 in shallow
    transformer_encoder = TransformerEncoder(encoder_layers, 2).cuda()  # 2 layers in EEGNet, 4 in shallow
    return transformer_encoder


def projector_t(args):
    projector_t = nn.Sequential(
        nn.Linear(args.dim_p, args.projector_dim1),  # TODO multiply configs.input_channels
        nn.BatchNorm1d(args.projector_dim1),
        nn.ReLU(),
        nn.Linear(args.projector_dim1, args.projector_dim2)).cuda()
    return projector_t


def encoder_f(args):
    encoder_layers = TransformerEncoderLayer(args.dim_e, dim_feedforward=2 * args.dim_e,
                                             nhead=2, )  # 2 heads in EEGNet, 5 in shallow
    transformer_encoder = TransformerEncoder(encoder_layers, 2).cuda()  # 2 layers in EEGNet, 4 in shallow
    return transformer_encoder


def projector_f(args):
    projector_t = nn.Sequential(
        nn.Linear(args.dim_p, args.projector_dim1),  # TODO multiply configs.input_channels
        nn.BatchNorm1d(args.projector_dim1),
        nn.ReLU(),
        nn.Linear(args.projector_dim1, args.projector_dim2)).cuda()
    return projector_t


def encoder_s(args):
    encoder_layers = TransformerEncoderLayer(args.dim_e, dim_feedforward=2 * args.dim_e,
                                             nhead=2, )  # 2 heads in EEGNet, 5 in shallow
    transformer_encoder = TransformerEncoder(encoder_layers, 2).cuda()  # 2 layers in EEGNet, 4 in shallow
    return transformer_encoder


def projector_s(args):
    projector_t = nn.Sequential(
        nn.Linear(args.dim_p, args.projector_dim1),  # TODO multiply configs.input_channels
        nn.BatchNorm1d(args.projector_dim1),
        nn.ReLU(),
        nn.Linear(args.projector_dim1, args.projector_dim2)).cuda()
    return projector_t


def projector_onelayer(args):
    projector_t = nn.Linear(args.feature_deep_dim, args.projector_dim2).cuda()
    return projector_t


# dynamic change the weight of the domain-discriminator
def calc_coeff(iter_num, alpha=10.0, max_iter=10000.0):
    return float(2.0 / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Net_ln2(nn.Module):
    def __init__(self, n_feature, n_hidden, bottleneck_dim):
        super(Net_ln2, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, bottleneck_dim)
        self.fc2.apply(init_weights)
        self.ln2 = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        x = x.view(x.size(0), -1)
        return x


class Net_CFE(nn.Module):
    def __init__(self, input_dim=310, bottleneck_dim=64):
        if input_dim < 256:
            print('\nwarning', 'input_dim < 256')
        super(Net_CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_dim, 256),
            # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, bottleneck_dim),  # default 64
            # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, hidden_dim, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(hidden_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(hidden_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_classifier_xy(nn.Module):
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier_xy, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        y = self.fc(x)
        return x, y


class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(tr.tensor(1.) * init_weights)

    def forward(self, x):
        x = self.w * tr.ones((x.shape[0]), 1).cuda()
        x = tr.sigmoid(x)
        return x


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class Discriminator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ln1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(self.bn(x))
        y = tr.sigmoid(x)
        return y


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size1, hidden_size2):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size1)
        self.ad_layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.ad_layer3 = nn.Linear(hidden_size2, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


class EEGWaveNet(nn.Module):
    def __init__(self, n_chans, n_classes):
        super(EEGWaveNet, self).__init__()

        self.temp_conv1 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv2 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv3 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv4 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv5 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv6 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)

        self.chpool1 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool2 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool3 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool4 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

        self.chpool5 = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01))

    def forward(self, x, return_feature=False):
        x = x.squeeze(2)
        temp_x = self.temp_conv1(x)
        temp_w1 = self.temp_conv2(temp_x)
        temp_w2 = self.temp_conv3(temp_w1)
        temp_w3 = self.temp_conv4(temp_w2)
        temp_w4 = self.temp_conv5(temp_w3)
        temp_w5 = self.temp_conv6(temp_w4)

        w1 = self.chpool1(temp_w1).mean(dim=(-1))
        w2 = self.chpool2(temp_w2).mean(dim=(-1))
        w3 = self.chpool3(temp_w3).mean(dim=(-1))
        w4 = self.chpool4(temp_w4).mean(dim=(-1))
        w5 = self.chpool5(temp_w5).mean(dim=(-1))

        concat_vector = tr.cat([w1, w2, w3, w4, w5], 1)
        return concat_vector
        # classes = self.classifier(concat_vector)
        #
        # return classes
