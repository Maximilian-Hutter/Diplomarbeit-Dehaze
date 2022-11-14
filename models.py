# combine base model: https://arxiv.org/pdf/2111.09733v1.pdf with large kernel structure https://arxiv.org/pdf/2209.01788v1.pdf
# Shallow Layers using the Large Kernel: use LKD before SHA in the MHA
# see what happens when you reaplce the attention in CoT with the DLKCB
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_models import *

class CoT(nn.Module): # https://arxiv.org/pdf/2107.12292v1.pdf change BN to IN
    def __init__(self) -> None:
        super(CoT, self).__init__()

    def forward(self,x):
        return out

class TailModule(nn.Module):
    def __init__(self, in_feat, out_feat, kernel, padw, padh):
        super(TailModule, self).__init__()
        pad = (padw,padw,padh,padh)
        self.padding = nn.ReflectionPad2d(pad)
        self.dlkcb = ConvBlock(in_feat, out_feat, kernel)
        self.elu = nn.ELU()
        self.conv2 = ConvBlock(out_feat, out_feat, 3)
        self.tanh = nn.Tanh()
    
    def forward(self,x):
        x = self.padding(x)
        x = self.dlkcb(x)
        x = self.elu(x)
        x = self.conv2(x)
        out = self.tanh(x)

        return out

class MHA(nn.Module):
    def __init__(self, in_feat, out_feat, num_parallel_conv, kernel_list, pad_list, groups):
        super(MHA, self).__init__()
        self.num_parallel_conv = num_parallel_conv
        self.kernel_list = kernel_list
        self.pad_list = pad_list
        self.parallel_conv = []
        
        for i,_ in enumerate(num_parallel_conv, start = 0):
            kernel = kernel_list[i]
            pad = pad_list[i]
            dlkcb = DLKCB(in_feat, out_feat, kernel, pad=pad)
            dlkcb.cuda()
            self.parallel_conv.append(dlkcb)

        self.lrelu = nn.LeakyReLU()
        self.convsha = ConvBlock(in_feat, out_feat, pad=1)
        self.sha = SHA(in_feat, out_feat, groups)

    def forward(self,x):
        res = x
        par_out = x
        for i in self.num_parallel_conv:
            conv = self.parallel_conv[i]
            par_out = conv(par_out)
            x = torch.add(par_out,x)

        x = self.lrelu(x)
        x = self.convsha(x)
        x = self.sha(x)
        out = torch.add(res,x)

        return out

class MHAC(nn.Module):
    def __init__(self) -> None:
        super(MHAC, self).__init__()

        self.mha = MHA()
        self.cot = CoT()
        self.aff = AdaptiveFeatureFusion()

    def forward(self,x):

        mhaout = self.mha(x)
        cotout = self.cot(x)

        out = self.aff(mhaout, cotout)
    
        return out
        
class SHA(nn.Module):
    def __init__(self, in_feat, out_feat, groups, kernel = 3):
        super(SHA,self).__init__()
        self.groups = groups

        # might be wrong
        self.avgh = nn.AvgPool2d((kernel,1),stride=1) # kernel of size 1 horizontaly and 0 verticaly
        self.maxh = nn.MaxPool2d((kernel,1),stride=1)

        self.avgv = nn.AvgPool2d((1,kernel),stride=1)
        self.maxv = nn.MaxPool2d((1,kernel),stride=1)
        #

        self.shuffle = nn.ChannelShuffle(groups)

        self.relu6 = nn.ReLU6()

        self.conv1 = ConvBlock(in_feat,out_feat, pad=1)
        self.conv2 = ConvBlock(in_feat,out_feat, pad=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        print(x.shape)
        res = x
        havg = self.avgh(x)
        hmax = self.maxh(x)
        h = torch.add(havg, hmax)
   

        vavg = self.avgv(x)
        vmax = self.maxv(x)
        v = torch.add(vavg, vmax)


        h = F.pad(h, (0,0,1,1), "constant",0)
        v = F.pad(v, (1,1), "constant",0)
        print(h.shape)
        print(v.shape)
        x = torch.cat((h,v))
        # put x to cpu because channel_shuffle no cuda backend
        x = x.to("cpu")
        x = self.shuffle(x) # cuda error
        x = x.to("cuda")

        # put cuda back to cpu
        x = self.conv1(x)
        x = self.relu6(x)

        x = torch.split(x,2)
        x1 = x[0]
        x2 = x[1]

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        
        out = torch.mul(x1,x2)
        out = self.sigmoid(out)

        out = torch.mul(out,res)


        return out

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, in_feat, inner_feat, kernel, groups):
        super(AdaptiveFeatureFusion, self).__init__()

        self.dlkcb = DLKCB(in_feat, inner_feat, kernel)
        self.elu = nn.ELU()
        self.sha = SHA(inner_feat,inner_feat,groups)
        self.conv1 = ConvBlock(inner_feat, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, y):

        x = torch.cat(x,y)
        x = self.dlkcb(x)
        x = self.elu(x)
        x = self.sha(x)
        x = self.conv1(x)
        out = self.sigmoid(x)
        
        return out

class DensityEstimation(nn.Module):
    def __init__(self, in_feat, kernel, groups, padw, padh, first_conv_feat = 64):
        super(DensityEstimation, self).__init__()

        # path # conv -> reflective pad, -> sha -> conv -> sigmoid
        self.dlkcb = DLKCB(in_feat, first_conv_feat, kernel)
        self.pad = nn.ReflectionPad2d((padw,padw,padh,padh))
        self.sha = SHA(first_conv_feat, first_conv_feat, groups)
        self.conv = ConvBlock(first_conv_feat, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        res = x
        x = self.dlkcb(x)
        x = self.pad(x)
        x = self.sha(x)
        x = self.conv(x)
        out = self.sigmoid(x)

        out = torch.mul(out, res)

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