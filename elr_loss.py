import torch.nn.functional as F
import torch
import torch.nn as nn


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.7):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        
    def forward(self, index, output, label):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + 3*elr_reg
        return  final_loss