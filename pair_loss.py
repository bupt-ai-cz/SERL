import torch
import torch.nn as nn


class Pair_loss(nn.Module):
    def __init__(self):
        super(Pair_loss, self).__init__()

    def forward(self, gt1, gt2, mask1, mask2, prob_1, prob_2):
        target_ulb = pairwise_target(gt1, gt2, mask1, mask2)
        loss = advbce_unlabeled(target_ulb, prob_1, prob_2, bce=BCE_softlabels)
        return loss


def BCE_softlabels(prob1, prob2, simi):
    P = prob1.mul_(prob2)
    P = P.sum(1)
    loss = - (simi * torch.log(P + 1e-7) + (1. - simi) * torch.log(1. - P + 1e-7))
    return loss.mean()


def advbce_unlabeled(target_ulb, prob1, prob2, bce):
    prob_bottleneck_row, prob_bottleneck_col = PairEnum2D_cross(prob1, prob2)
    adv_bce_loss = bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
    return adv_bce_loss


def pairwise_target(target1, target2, mask1, mask2):
    target_row1, target_col2 = PairEnum1D_cross(target1, target2)
    mask_row1, mask_col2 = PairEnum1D_cross(mask1, mask2)
    target_ulb = torch.zeros(target1.size(0) * target2.size(0)).float().cuda()
    target_ulb[(target_row1==target_col2)&(mask_row1.eq(1.))&(mask_col2.eq(1.))] = 1
    return target_ulb


def PairEnum1D_cross(x1_, x2_):
    x1 = x1_.repeat(x2_.size(0), )
    x2 = x2_.repeat(x1_.size(0)).view(-1, x2_.size(0)).transpose(1, 0).reshape(-1)
    return x1, x2


def PairEnum2D_cross(x1_, x2_):
    x1 = x1_.repeat(x2_.size(0), 1)
    x2 = x2_.repeat(1, x1_.size(0)).view(-1, x1_.size(1))
    return x1, x2