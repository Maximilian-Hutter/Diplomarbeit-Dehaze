import torch.nn as nn
import torch

class DLKCB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride = 1, pad = 0):
        super(DLKCB, self).__init__()
        
        self.conv1 = nn.Conv2d(in_feat, in_feat, 1)
        self.dwconv = nn.Conv2d(in_feat, in_feat, kernel,stride, pad, groups=in_feat)
        self.dwdconv = nn.Conv2d(in_feat, in_feat, kernel, stride, pad, dilation=kernel, groups=in_feat)
        self.conv2 = nn.Conv2d(in_feat, out_feat, 1)

    def forward(self,x):

        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.dwdconv(x)
        out = self.conv2(x)

        return out

class CEFN(nn.Module):
    def __init__(self,feat,pool_kernel,pool_stride, shape):
        super(CEFN, self).__init__()

        shape = shape[0] * shape[1] * shape[2]
        small_shape = shape / 2
        self.norm1 = nn.InstanceNorm2d(feat)

        self.linear = nn.Linear(shape,shape)
        self.dwconv = nn.Conv2d(feat, feat,3,stride=1,padding=1, groups=feat)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(shape,shape)
        self.norm2 = nn.InstanceNorm2d(feat)

        self.pool = nn.AvgPool2d(pool_kernel,pool_stride)
        self.linear3 = nn.Linear(small_shape,small_shape)
        self.relu2 = nn.ReLU()
        self.linear4 = nn.Linear(small_shape,small_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        res = x
        x = self.norm1(x)

        x = self.linear(x)
        x = self.dwconv(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.norm2(x)

        x2 = self.pool(x)
        x2 = self.linear3(x2)
        x2 = self.relu2(x2)
        x2 = self.linear4(x2)
        x2 = self.sigmoid(x2)

        out = torch.mul(x,x2)
        out = torch.add(out,res)

        return out
        
class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat,kernel_size = 3, stride = 1, pad = 1, dilation = 1, groups = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad, dilation, groups)

    def forward(self,x):

        out = self.conv(x)
        return out

class DepthWiseConv(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride,pad, dilation):
        super().__init__()

        self.depth_conv = nn.Conv2d(in_feat, in_feat, kernel_size, stride, pad, dilation, groups=in_feat)
        self.point_conv = nn.Conv2d(in_feat, out_feat, 1)
    
    def forward(self,x):
        x = self.depth_conv(x)
        out = self.point_conv(x)
        return out