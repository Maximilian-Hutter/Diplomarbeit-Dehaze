import torch.nn as nn
import torch

class DLKCB(nn.Module):
    def __init__(self):
        super(DLKCB, self).__init__()
        
    def forward(self,x):
        return out

class CEFN(nn.Module):
    def __init__(self):
        super(CEFN, self).__init__()

    def forward(self,x):
        return out
        
class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat,kernel_size = 3, stride = 1, pad = 1, dilation = 0, groups = 1, activation = "ReLU", use_cefn = True, ):
        super().__init__()

        self.use_cefn = use_cefn

        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad, dilation, groups)
        self.cefn = CEFN()
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()
        self.leakyrelu = nn.LeakyReLU()
        self.rrelu = nn.RReLU()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.prelu = nn.PReLU()
        self.silu = nn.SiLU()
    def forward(self,x):
        if self.use_cefn:
            out = self.cefn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation == "ReLU":
            out = self.relu(out)
        elif self.actvation == "SeLU":
            out = self.selu(out)
        elif self.activation == "LReLU":
            out = self.leakyrelu(out)
        elif self.activation == "Sigmoid":
            out = self.sigmoid(out)
        elif self.activation == "ReLU6":
            out = self.relu6(out)
        elif self.activation == "RReLU":
            out = self.rrelu(out)
        elif self.activation == "PReLU":
            out = self.prelu(out)
        elif self.activation == "SiLU":
            out = self.silu(out)
        else:
            out = out

        return out