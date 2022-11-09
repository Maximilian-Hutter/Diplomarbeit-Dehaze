# combine base model: https://arxiv.org/pdf/2111.09733v1.pdf with large kernel structure https://arxiv.org/pdf/2209.01788v1.pdf
# Shallow Layers using the Large Kernel: use LKD before SHA in the MHA
# see what happens when you reaplce the attention in CoT with the DLKCB
from numpy import outer
import torch
import torch.nn as nn
from basic_models import *

class CoT(nn.Module): # https://arxiv.org/pdf/2107.12292v1.pdf change BN to IN
    def __init__(self) -> None:
        super(CoT, self).__init__()

    def forward(self,x):
        return out

class TailModule(nn.Module):
    def __init__(self):
        super(TailModule, self).__init__()
    
    def forward(self,x):
        return out

class MHA(nn.Module):
    def __init__(self):
        super(MHA, self).__init__()
        self.parallel_conv = []
        
        for i in num_parallel_conv:
            kernel = kernel_list[i]
            pad = pad_list[i]
            self.parallel_conv.append(DLKCB(in_feat, out_feat, kernel, pad=pad))

        self.lrelu = nn.LeakyReLU()
        self.convsha = ConvBlock()

    def forward(self,x):
        res = x

        for i in self.num_parallel_conv:
            par_out = self.parallel_conv[i]
            x = torch.add(par_out,x)

        x = self.lrelu(x)
        x = self.convsha(x)
        x = SHA(x)
        out = torch.add(res,x)

        return out

class MHAC(nn.Module):
    def __init__(self) -> None:
        super(MHAC, self).__init__()
    
    def forward(self,x):
        return out
        
class SHA(nn.Module):
    def __init__(self):
        super(SHA,self).__init__()

        # might be wrong
        self.avgh = nn.AvgPool2d((1,0)) # kernel of size 1 horizontaly and 0 verticaly
        self.maxh = nn.MaxPool2d((1,0))

        self.avgv = nn.AvgPool2d((0,1))
        self.maxv = nn.MaxPool2d((0,1))

        self.shuffle = nn.ChannelShuffle()

        self.relu6 = nn.ReLU6()

        self.conv1 = ConvBlock()
        self.conv2 = ConvBlock()

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        res = x
        havg = self.avgh(x)
        hmax = self.maxh(x)
        h = torch.add(havg, hmax)

        vavg = self.avgv(x)
        vmax = self.maxv(x)
        v = torch.add(vavg, vmax)

        x = torch.cat((h,v))

        x = self.shuffle(x)
        x = self.conv1(x)
        x = self.relu6(x)

        x = torch.split(x)
        x1 = x[0]
        x2 = x[1]

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        
        out = torch.mul(x1,x2)
        out = self.sigmoid(out)
        out = torch.mul(out,res)

        return out

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self):
        super(AdaptiveFeatureFusion, self).__init__()

    def forward(self,x):
        return out

class DensityEstimation(nn.Module):
    def __init__(self):
        super(DensityEstimation, self).__init__()

    def forward(self,x):
        return out

class Shallow(nn.Module):
    def __init__(self):
        super(Shallow, self).__init__()

    def forward(self,x):
        return out

class Deep(nn.Module):
    def __init__(self):
        super(Deep,self).__init__()
    
    def forward(self,x):
        return out

class Dehaze(nn.Module):
    def __init__(self):
        super(Dehaze, self).__init__()

    def forward(self, x):
        return out