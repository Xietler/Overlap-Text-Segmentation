# -*- coding: utf-8 -*-
# @Time    : 2019/2/24 16:30
# @Author  : Ruichen Shao
# @File    : focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np

# positive samples and negtive samples are not balanced
# so we need to add class weight in loss calculation
def calc_weights(dataset=None, dataloader=None, num_classes=None, save_path=None):
    if os.path.exists(save_path):
        return np.load(save_path)
    else:
        z = np.zeros((num_classes,))
        print('Calculating classes weights')
        with tqdm(dataloader) as t:
            for sample in t:
                _, _, _, mask = sample
                mask = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
                count_l = np.bincount(mask)
                z += count_l
        total_frequency = np.sum(z)
        class_weights = []
        for f in z:
            class_weight = 1 / (np.log(1.02 + (f / total_frequency)))
            class_weights.append(class_weight)
        ret = np.array(class_weights)
        ret = ret / ret.sum() * num_classes
        np.save(save_path, ret)
        print(ret)
        return ret

# focal loss for multi labels
class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='sum', balance_param=0.001):
        super(FocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        class_num = len(weight)
        # weight = np.repeat(weight, 400*400)
        # self.weight = weight.reshape((class_num, 400, 400)).double().cuda()
        self.weight = weight.double().cuda()

        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # input and target should be (N, C)

        logpt = -F.binary_cross_entropy_with_logits(input.double(), target.double(), pos_weight=self.weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt
        return self.balance_param * focal_loss

if __name__ == '__main__':
    from segmentation.dataset.Dataset import DigitDataset
    dataset1 = DigitDataset('C:/Users/SRC/Downloads/SegDataset/number_segment_1/img',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_1/label',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_1/label.txt',
                           offset=0)
    dataset2 = DigitDataset('C:/Users/SRC/Downloads/SegDataset/number_segment_2/img',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_2/label',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_2/label.txt',
                           offset=25000)
    from torch.utils.data import DataLoader, ConcatDataset

    dataset = ConcatDataset([dataset1, dataset2])
    dataloader = DataLoader(dataset)
    calc_weights(dataset, dataloader, 11, 'C:/Users/SRC/PycharmProjects/DigitSegmentation/segmentation/dataset/class_weights.npy')