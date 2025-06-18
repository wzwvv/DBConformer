import torch.nn as nn
import torch

class FC(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        x = self.fc(x)
        return x

class FC_wave(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_wave, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(nn_in, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, nn_out))
    def forward(self, x):
        x = self.fc(x)
        return x

class FC_xy(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_xy, self).__init__()
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        y = self.fc(x)
        return x, y

class FC_xy_2l(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_xy_2l, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(nn_in, 64),
            nn.Linear(64, 32),
            nn.Linear(32, nn_out))

    def forward(self, x):
        y = self.fc(x)
        return x, y


class FC_wave_xy(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_wave_xy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(nn_in, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, nn_out))
    def forward(self, x):
        y = self.fc(x)
        return x, y