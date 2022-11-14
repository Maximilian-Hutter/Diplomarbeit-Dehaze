from models import *
from basic_models import *
import numpy as np
import torch
import torch.nn as nn
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary

hparams = {
    "seed": 123,
    "gpus": 1,
    "height":None,
    "width":None,
}

np.random.seed(hparams["seed"])    # set seed to default 123 or opt
torch.manual_seed(hparams["seed"])
torch.cuda.manual_seed(hparams["seed"])
gpus_list = range(hparams["gpus"])
hostname = str(socket.gethostname)
cudnn.benchmark = True

# defining shapes

#Net = MHA(in_feat= 3, out_feat = 3, num_parallel_conv=range(3), kernel_list=[3,5,7,9], pad_list=[2,6,12,20], groups=3).cuda()
Net = MHAC(64, 64, range(3), [3,5,7], [2,6,12], 4).cuda()


summary(Net, (64, 128, 128))

# pytorch_params = sum(p.numel() for p in Net.parameters())
# print("Network parameters: {}".format(pytorch_params))

# def print_network(net):
#     num_params = 0
#     for param in net.parameters():
#         num_params += param.numel()
#     print(net)
#     print('Total number of parameters: %d' % num_params)

# print('===> Building Model ')
# Net = Net


# print('----------------Network architecture----------------')
# print_network(Net)
# print('----------------------------------------------------')