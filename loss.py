import torch
import torch.nn as nn

class ChabonnierLoss(nn.Module):
    def __init__(self, eps):
        super().__init__()

        self.eps = eps
    
    def forward(self, gen, label):
        b, c, h, w = label.size()
        loss = torch.sqrt(torch.add(torch.abs(torch.pow(gen,2) - torch.pow(label,2)), pow(self.eps,2)))
        loss = loss/(c*b*h*w)
        return torch.sum(loss)