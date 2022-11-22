import torch.nn as nn
import torch

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self,x):
        batch, channels, height, width = x.size()
        assert (channels % self.groups == 0)
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(batch, channels, height, width)
        return out

class DLKCB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride = 1, pad = 4):
        super(DLKCB, self).__init__()
        
        self.pad = nn.ReflectionPad2d((pad, pad, pad, pad))
        self.conv1 = nn.Conv2d(in_feat, in_feat, 1)
        self.shuffle = ChannelShuffle(in_feat // 4)
        self.dwconv = nn.Conv2d(in_feat, in_feat, kernel,stride, groups=in_feat)
        self.dwdconv = nn.Conv2d(in_feat, in_feat, kernel, stride, dilation=kernel, groups=in_feat)
        self.conv2 = nn.Conv2d(in_feat, out_feat, 1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pad(x)
        x = self.dwconv(x)
        x = self.dwdconv(x)
        x = self.conv2(x)

        return x

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

        x = torch.mul(x,x2)
        x = torch.add(x,res)

        return x
        
class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat,kernel_size = 3, stride = 1, pad = 1, dilation = 1, groups = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad, dilation, groups)

    def forward(self,x):

        x = self.conv(x)
        return x

class DepthWiseConv(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, stride,pad, dilation):
        super().__init__()

        self.depth_conv = nn.Conv2d(in_feat, in_feat, kernel_size, stride, pad, dilation, groups=in_feat)
        self.point_conv = nn.Conv2d(in_feat, out_feat, 1)
    
    def forward(self,x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class TransposedUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, kernel = 11, stride = 2, use_dlkcb = True):
        super().__init__()
        self.use_dlkcb = use_dlkcb
    
        self.dlkcb = DLKCB(in_feat, out_feat,kernel, pad=60)    # if weird stuff happens disable

        if use_dlkcb is False:
            padding = 2
        self.up = nn.ConvTranspose2d(out_feat, out_feat, 2, stride, padding = 0)

    def forward(self,x):

        if self.use_dlkcb is True:
            x = self.dlkcb(x)

        x = self.up(x)
        #print(out.shape)

        return x