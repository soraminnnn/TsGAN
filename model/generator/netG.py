import torch.nn as nn
from model.generator.condition_BN import CategoricalConditionalBatchNorm1d
import torch
from torch.nn import init
import math

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, n_classes=0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)

        if n_classes > 0:
            self.bn1 = CategoricalConditionalBatchNorm1d(n_classes, n_inputs)
            self.bn2 = CategoricalConditionalBatchNorm1d(n_classes, n_outputs)
        else:
            self.bn1 = nn.BatchNorm1d(n_inputs)
            self.bn2 = nn.BatchNorm1d(n_outputs)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def short_cut(self,x):
        res = x if self.downsample is None else self.downsample(x)
        return res

    def residual(self, x, y=None):
        h = x
        if y is not None:
            h = self.bn1(h,y)
            h = self.relu(h)
            h = self.conv1(h)
            h = self.chomp1(h)
            h = self.bn2(h,y)
            h = self.relu(h)
            h = self.conv2(h)
            h = self.chomp2(h)
        else:
            h = self.bn1(h)
            h = self.relu(h)
            h = self.conv1(h)
            h = self.chomp1(h)
            h = self.bn2(h)
            h = self.relu(h)
            h = self.conv2(h)
            h = self.chomp2(h)
        return h

    def forward(self, x, y=None):
        return self.short_cut(x) + self.residual(x,y)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, n_classes):
        super(TemporalConvNet, self).__init__()
        num_levels = len(num_channels)
        self.network = nn.ModuleList()
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.network.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, n_classes=n_classes))

    def forward(self, x, y=None):
        h = x
        for model in self.network:
            h = model(h,y)
        return h

class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, input_dim, output_dim, num_channels, kernel_size, n_classes=0):
        super(Generator, self).__init__()
        self.input = nn.Linear(latent_dim,seq_len*input_dim)
        self.input_dim = input_dim
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, n_classes=n_classes)
        self.output = nn.Sequential(
            nn.Conv1d(num_channels[-1],output_dim,kernel_size=1,padding=0,stride=1),
            nn.Tanh()
        )

    def forward(self, x, y=None):
        h = self.input(x).view(x.shape[0],self.input_dim,-1)
        h = self.tcn(h,y)
        h = self.output(h)
        return h

