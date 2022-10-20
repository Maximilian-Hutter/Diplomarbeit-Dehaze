# combine base model: https://arxiv.org/pdf/2111.09733v1.pdf with large kernel structure https://arxiv.org/pdf/2209.01788v1.pdf
import torch
import torch.nn as nn

class dehaze(nn.Module):
    def __init__(self):
        super(dehaze, self).__init__()

    def forward(self, x):
        return out