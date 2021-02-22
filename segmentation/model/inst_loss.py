# -*- coding: utf-8 -*-
# @Time    : 2019/3/4 21:26
# @Author  : Ruichen Shao
# @File    : inst_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# loss for instance num and location
class InstLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, reduction='mean', balance_param=0.05, ratio=20):
        super(InstLoss, self).__init__()

        self.reduction = reduction
        self.balance_param = balance_param
        self.ratio = ratio

    def forward(self, input_cls, target_cls, input_loc, target_loc, weight):
        # transform to double tensor
        input_cls, target_cls, input_loc, target_loc, weight = input_cls.double(), target_cls.double(), \
                                                               input_loc.double(), target_loc.double(), weight.double()
        # location loss first
        loss_loc = self._smooth_l1_loss(input_loc, target_loc, weight)

        # cls num loss second
        loss_cls = F.mse_loss(input_cls, target_cls, reduction=self.reduction)
        # print(loss_loc, loss_cls)
        return self.balance_param * (self.ratio * loss_loc + loss_cls)

    def _smooth_l1_loss(self, input, target, weight, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        diff = input - target
        diff_abs = torch.abs(diff)
        smoothL1_sign = (diff_abs < 1. / sigma_2).detach().double()
        loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
               + (diff_abs - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        loss = weight * loss
        for i in sorted(dim, reverse=True):
            loss = loss.sum(i)
        loss = loss.mean()
        return loss
