# combine base model: https://arxiv.org/pdf/2111.09733v1.pdf with large kernel structure https://arxiv.org/pdf/2209.01788v1.pdf
# Shallow Layers using the Large Kernel: use LKD before SHA in the MHA
# see what happens when you reaplce the attention in CoT with the DLKCB
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_models import *
# from fightingcv_attention.attention.CoTAttention import *

class CoT(nn.Module):
    def __init__(self, in_feat=512,kernel=3):
        super().__init__()
        self.in_feat=in_feat
        self.kernel_size=kernel

        self.key_embed=nn.Sequential(
            nn.Conv2d(in_feat,in_feat,kernel_size=kernel,padding=kernel//2,groups=4,bias=False),
            nn.InstanceNorm2d(in_feat),
            nn.ELU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(in_feat,in_feat,1,bias=False),
            nn.InstanceNorm2d(in_feat)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*in_feat,2*in_feat//factor,1,bias=False),
            nn.InstanceNorm2d(2*in_feat//factor), # BN to IN
            nn.ELU(), # ReLU to ELU
            nn.Conv2d(2*in_feat//factor,kernel*kernel*in_feat,1)
        )


    def forward(self, x):   # modified from xmu-xiaoma666/External-Attention-pytorch

        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat((k1,x),dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)

        return k1+k2


class TailModule(nn.Module):
    def __init__(self, in_feat, out_feat, kernel, padw, padh):
        super(TailModule, self).__init__()
        pad = (padw,padw,padh,padh)
        self.padding = nn.ReflectionPad2d(pad)
        self.conv1 = ConvBlock(in_feat, out_feat, 3)
        self.elu = nn.ELU()
        self.conv2 = ConvBlock(out_feat, out_feat, 3)
        self.tanh = nn.Tanh()
    
    def forward(self,x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.tanh(x)

        return x

class MHA(nn.Module):
    def __init__(self, in_feat, out_feat, num_parallel_conv, kernel_list, pad_list, groups):
        super(MHA, self).__init__()
        self.num_parallel_conv = num_parallel_conv
        self.kernel_list = kernel_list
        self.pad_list = pad_list
        self.parallel_conv = []
        
        # for i,_ in enumerate(range(num_parallel_conv), start = 0):
        #     kernel = kernel_list[i]
        #     pad = pad_list[i]
        #     dlkcb = DLKCB(in_feat, out_feat, kernel, pad=pad)
        #     dlkcb.cuda()
        #     self.parallel_conv.append(dlkcb)

        self.dlkcb = DLKCB(in_feat, in_feat, kernel = kernel_list[0], pad = pad_list[0])
        self.dlkcb2 = DLKCB(in_feat, in_feat, kernel = kernel_list[1], pad = pad_list[1])

        self.lrelu = nn.LeakyReLU()
        self.convsha = ConvBlock(in_feat, out_feat, pad=1)
        self.sha = SHA(in_feat, out_feat, groups)

    def forward(self,x):
        res = x
        
        # for i in range(self.num_parallel_conv):

        #     conv = self.parallel_conv[i]
        #     par_out = conv(par_out)
        #     x = torch.add(par_out,x)

        x1 = self.dlkcb(x)
        x2 = self.dlkcb2(x)

        x = torch.add(x,x1)
        x = torch.add(x, x2)

        

        x = self.lrelu(x)
        x = self.convsha(x)
        x = self.sha(x)
        x = torch.add(res,x)

        return x

class MHAC(nn.Module):
    def __init__(self, in_feat, inner_feat, out_feat, num_parallel_conv, kernel_list, pad_list, groups, kernel = 3):
        super(MHAC, self).__init__()

        self.mha = MHA(in_feat, in_feat, num_parallel_conv, kernel_list, pad_list, groups)
        self.cot = CoT(in_feat)
        self.aff = AdaptiveFeatureFusion()

    def forward(self,x):

        mhaout = self.mha(x)
        cotout = self.cot(x)
        x = self.aff(mhaout, cotout)
        
    
        return x
        
class SHA(nn.Module):
    def __init__(self, in_feat, out_feat, groups, kernel = 3, downsample = False):
        super(SHA,self).__init__()
        self.groups = groups
        self.in_feat = in_feat
        self.out_feat = out_feat

        # might be wrong
        self.avgh = nn.AvgPool2d((kernel,1),stride=1) # kernel of size 1 horizontaly and 0 verticaly
        self.maxh = nn.MaxPool2d((kernel,1),stride=1)

        self.avgv = nn.AvgPool2d((1,kernel),stride=1)
        self.maxv = nn.MaxPool2d((1,kernel),stride=1)
        #

        self.shuffle = ChannelShuffle(groups)

        self.relu6 = nn.ReLU6()
        self.downsample = downsample
        if downsample is True:
            stride = 2
        else: 
            stride = 1

        self.conv1 = ConvBlock(in_feat*2,in_feat*2, pad=1)
        self.conv2 = ConvBlock(in_feat,out_feat, stride = stride, pad=1)

        self.sigmoid = nn.Sigmoid()

        self.convres = ConvBlock(in_feat, out_feat)
        self.down = nn.Upsample(scale_factor=0.5)
    def forward(self,x):
        res = x
        
        havg = self.avgh(x)
        hmax = self.maxh(x)
        h = torch.add(havg, hmax)
   

        vavg = self.avgv(x)
        vmax = self.maxv(x)
        v = torch.add(vavg, vmax)


        h = F.pad(h, (0,0,1,1), "constant",0)
        v = F.pad(v, (1,1), "constant",0)

        x = torch.cat((h,v), dim= 1)
        x = self.shuffle(x)
        
        x = self.conv1(x)
        x = self.relu6(x)
        x = torch.split(x,self.in_feat, dim = 1)

        x1 = x[0]
        x2 = x[1]

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        
        x = torch.mul(x1,x2)
        
        x = self.sigmoid(x)

        if self.downsample is True:
            res = self.down(res)

        if self.in_feat is not self.out_feat:
            res = self.convres(x)

        x = torch.mul(x,res)

        return x

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self):
        super(AdaptiveFeatureFusion, self).__init__()
        # image was of density estimation

        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

    def forward(self,x, y):

        x = torch.add(self.sig1(x), self.sig2(y))
        
        return x

class DensityEstimation(nn.Module):
    def __init__(self, in_feat, kernel, groups, padw, padh, first_conv_feat = 64):
        super(DensityEstimation, self).__init__()

        # path # conv -> reflective pad, -> sha -> conv -> sigmoid
        self.dlkcb = DLKCB(in_feat, first_conv_feat, kernel)
        self.elu = nn.ELU()
        self.pad = nn.ReflectionPad2d((padw,padw,padh,padh))
        self.sha = SHA(first_conv_feat, first_conv_feat, groups)
        self.conv1 = ConvBlock(first_conv_feat, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, y):
        res = x
        x = torch.cat((x,y), dim = 1) # might be wrong dim
        x = self.dlkcb(x)
        x = self.elu(x)
        x = self.sha(x)
        x = self.conv1(x)
        x = self.sigmoid(x)

        x = torch.mul(x, res)

        return x

class Shallow(nn.Module):
    def __init__(self, in_feat, inner_feat, num_mhac, num_parallel_conv, kernel_list, pad_list):
        super(Shallow, self).__init__()

        self.conv1 = ConvBlock(in_feat,in_feat)
        self.sha1 = SHA(in_feat, in_feat, in_feat, downsample=True)
        self.conv2 = ConvBlock(in_feat, inner_feat)
        self.sha2 = SHA(inner_feat, inner_feat, inner_feat//16, downsample=True)

        m = []
        for i in range(num_mhac):
            m.append(MHAC(inner_feat, inner_feat, inner_feat, num_parallel_conv, kernel_list, pad_list, inner_feat//16))

        self.mhacblock = nn.Sequential(*m)

        self.up1 = TransposedUpsample(inner_feat, inner_feat)
        self.sha3 = SHA(inner_feat, inner_feat, inner_feat//16)
        self.up2 = TransposedUpsample(inner_feat, in_feat)
        self.sha4 = SHA(in_feat, in_feat, in_feat)

        self.tail = TailModule(in_feat, in_feat, 3, 0, 0)

    def forward(self,x):
        res = x
        x = self.conv1(x)
        res1 = x
        x = self.sha1(x)
        x = self.conv2(x)
        res2 = x
        x = self.sha2(x)

        x = self.mhacblock(x)

        #print(x.shape)
        x = self.up1(x)
        #print(x.shape)
        #print(res2.shape)
        x = torch.add(x, res2)
        
        x = self.sha3(x)
        x = self.up2(x)
        #res1 = F.interpolate(res1, scale_factor=1.25)
        x = torch.add(x,res1)
        x = self.sha4(x)
        shares = x

        x = self.tail(x)

        #res = F.interpolate(res, scale_factor=1.25)
        x = torch.add(x, res)
        return x, shares

class Deep(nn.Module):
    def __init__(self, in_feat, inner_feat, out_feat, num_mhablock, num_parallel_conv, kernel_list, pad_list,down_deep):
        super(Deep,self).__init__()
        self.down_deep = down_deep
        if down_deep is True:
            stride = 2
        else:
            stride = 1
        self.downx = ConvBlock(in_feat, in_feat, stride=stride)
        self.downdense = ConvBlock(in_feat, in_feat, stride=stride)
        self.aff = AdaptiveFeatureFusion()
        self.conv1 = ConvBlock(in_feat, inner_feat)

        m = []
        for i in range(num_mhablock):
            m.append(MHA(inner_feat, inner_feat, num_parallel_conv, kernel_list, pad_list, 16))

        self.mhablocks = nn.Sequential(*m)

        self.up = TransposedUpsample(inner_feat, inner_feat)

        self.tail = TailModule(inner_feat, out_feat, 3, 0, 0)

    def forward(self,x, dense):

        x = self.downx(x)
        dense = self.downdense(dense)
        x = self.aff(x, dense)
        x = self.conv1(x)
        x = self.mhablocks(x)
        if self.down_deep is True:
            x = self.up(x)
        x = self.tail(x)

        return x

class Dehaze(nn.Module):
    def __init__(self, mhac_filter = 256, mha_filter = 16,num_mhablock = 10,num_mhac = 8, num_parallel_conv = 2, kernel_list = [3,5,7], pad_list = [4,12,24], down_deep = False,gpu_mode = True, scale_factor = 1):
        super(Dehaze, self).__init__()

        # if gpu_mode is True:
        #     self.shallow = Shallow(3, mhac_filter, num_mhac, num_parallel_conv, kernel_list, pad_list).to(torch.device("cuda:0")) # filter 256
        #     self.dense = DensityEstimation(6,3, 4, 0, 0).to(torch.device("cuda:0"))
        #     self.aff = AdaptiveFeatureFusion().to(torch.device("cuda:0"))
        #     self.deep = Deep(3, mha_filter, 3, num_mhablock, num_parallel_conv, kernel_list, pad_list).to(torch.device("cuda:0")) # filter 16
        # else:
        self.shallow = Shallow(3, mhac_filter, num_mhac, num_parallel_conv, kernel_list, pad_list) # filter 256
        self.dense = DensityEstimation(6,3, 4, 0, 0)
        self.aff = AdaptiveFeatureFusion()
        self.deep = Deep(3, mha_filter, 3, num_mhablock, num_parallel_conv, kernel_list, pad_list, down_deep) # filter 16

        self.scale_factor = scale_factor
        if scale_factor != 1:
            self.up = TransposedUpsample(3, 3, 11, scale_factor, False)
            
    def forward(self, hazy):

        pseudo, shares = self.shallow(hazy)
        density = self.dense(pseudo, hazy)
        density = torch.mul(density, shares)


        x = self.aff(pseudo, hazy)
        x = self.deep(x, density)

        if self.scale_factor != 1:
            x = self.up(x)
            pseudo = F.interpolate(pseudo, scale_factor=self.scale_factor)
            hazy = F.interpolate(hazy, scale_factor=self.scale_factor)

        x = torch.mul(x, pseudo)
        x = torch.add(x, hazy, alpha=0.5)

        return x, pseudo