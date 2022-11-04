# combine base model: https://arxiv.org/pdf/2111.09733v1.pdf with large kernel structure https://arxiv.org/pdf/2209.01788v1.pdf
# Shallow Layers using the Large Kernel: use LKD before SHA in the MHA
# instead of FN use CEFN
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

    def forward(self,x):
        res = x

        x1 = conv1(x)
        x2 = conv2(x)
        x = torch.add(x1, x2)
        x = torch.add(res,x)
        x = lrelu(x)
        x = convsha(x)
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

    def forward(self,x):
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