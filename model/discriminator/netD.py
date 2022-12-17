import torch.nn as nn
import torch.nn.utils as utils
import torch
import torch.nn.functional as F

class conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, downsample=False, sn=True):
        super(conv_Block, self).__init__()
        self.downsample = downsample
        if sn:
            self.conv = utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),n_power_iterations=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
        self.conv.weight.data.normal_(0, 0.01)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, x):
        h = self.activation(self.conv(x))
        if self.downsample:
            h = torch.nn.functional.avg_pool2d(h,2)
        return h

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_layers=3, downsample=[True]*3, sn=True):
        super(Discriminator, self).__init__()
        self.network = nn.ModuleList()
        channels=32
        for i in range(num_layers):
            if i==0:
                self.network.append(conv_Block(in_channels,channels,downsample=downsample[i],sn=sn))
            else:
                self.network.append(conv_Block(channels, channels*2, downsample=downsample[i], sn=sn))
                channels *= 2
        self.output = nn.Sequential(
            nn.Linear(channels,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = x
        for model in self.network:
            h = model(h)
        h = F.adaptive_avg_pool2d(h,(1,1)).view(h.shape[0],-1)
        h = self.output(h)
        return h

class CGAN_Discriminator(nn.Module):
    def __init__(self, in_channels, num_layers=3, downsample=[True]*3, sn=True, num_classes=0):
        super(CGAN_Discriminator, self).__init__()
        self.network = nn.ModuleList()
        channels=32
        for i in range(num_layers):
            if i==0:
                self.network.append(conv_Block(in_channels,channels,downsample=downsample[i],sn=sn))
            else:
                self.network.append(conv_Block(channels, channels*2, downsample=downsample[i], sn=sn))
                channels *= 2

        if num_classes > 0:
            self.l_y = nn.Embedding(num_classes,channels)  
        self.linear = nn.Linear(channels,1)

    def forward(self, x, y=None):
        h = x
        for model in self.network:
            h = model(h)
        h = F.adaptive_avg_pool2d(h,(1,1)).view(h.shape[0],-1)
        output = self.linear(h)
        if y is not None:
            output += torch.sum(self.l_y(y)*h, dim=1, keepdim=True)
        return torch.sigmoid(output)


