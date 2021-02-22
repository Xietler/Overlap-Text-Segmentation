# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 22:15
# @Author  : Ruichen Shao
# @File    : scheduler.py

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None, last_epoch=-1):
        self.last_epoch = last_epoch
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                # return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, metrics=None, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(metrics, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)