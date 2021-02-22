# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 12:56
# @Author  : Ruichen Shao
# @File    : metrics.py

import numpy as np
import torch

class Evaluator(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def calc_valid_iou(self, mask, mask_1, mask_2):
        # calculate iou without background
        mask = np.array(mask, dtype=np.int8)
        idx_h, idx_w = np.where(mask != 0)
        total = len(idx_h)
        mask = mask[idx_h, idx_w]
        mask_1 = mask_1[idx_h, idx_w]
        mask_2 = mask_2[idx_h, idx_w]
        intersection_1 = (mask_1 == mask).sum()
        intersection_2 = (mask_2 == mask).sum()

        iou = max(intersection_1 / total, intersection_2 / total)

        return iou

    def calc_iou(self, y_true, y_pred, threshold=0.5):
        # may not be good
        # wait to improve
        # input is (C, H, W)
        y_pred = torch.where(y_pred>threshold, torch.ones_like(y_pred), torch.zeros_like(y_pred))
        intersection = 0
        # this version maybe too strict
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[2]):
                if torch.equal(y_pred[:, i, j], y_true[:, i, j]) is True:
                    intersection += 1

        return intersection / (y_pred.shape[1] * y_pred.shape[2])

if __name__ == '__main__':
    eval = Evaluator(11)
    y_true = torch.rand(11, 400, 400)
    y_pred = torch.rand(11, 400, 400)
    print(eval.calc_iou(y_true, y_pred))



