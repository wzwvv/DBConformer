# --------------------------------------------------------
# copied from https://github.com/ravikiran-mane/FBCNet
# --------------------------------------------------------
import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
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

class LogVarLayer(nn.Module):
    '''
    计算数据在指定维度上的对数方差
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))


class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s.
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(nBands, m*nBands, (nChan//nBands, 1), groups=nBands,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(m * nBands),
            swish()
        )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            #nn.LogSoftmax(dim=1)
        )

    def __init__(self, data_name, nChan, nTime, nClass=2, nBands=2, m=32, temporalLayer='LogVarLayer', strideFactor=3, doWeightNorm=True, *args, **kwargs):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor
        self.data_name = data_name

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm=doWeightNorm)

        # Formulate the temporal agreegator
        self.temporalLayer = LogVarLayer(dim=3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m * self.nBands * self.strideFactor, nClass, doWeightNorm=doWeightNorm)

    def forward(self, x):
        # N, B, C, T
        if self.data_name == 'BNCI2015001' or self.data_name == 'MI1-7' or self.data_name == 'MI1':
            x = x[:, :-1, :]
        N, C, T = x.shape
        x = x.reshape(N, self.nBands, C//self.nBands, T)
        #x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.scb(x)  # [32, 224, 1, 1251]
        if 'BNCI2014001' in self.data_name or self.data_name == 'BNCI2014002' or self.data_name == 'BNCI2015001':
            x = x[:, :, :, :-2]
        elif 'BNCI2014004' in self.data_name:
            x = x[:, :, :, :-1]
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        fea = torch.flatten(x, start_dim=1)
        pred = self.lastLayer(fea)
        return fea, pred