# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 16:01
# @Author  : Ruichen Shao
# @File    : train.py

import sys

sys.path.insert(0, '/app/Yibao-Cup-code')
import argparse
import os
import numpy as np
from tqdm import tqdm

from segmentation.model.deeplab import DeepLab
from segmentation.model.focal_loss import FocalLoss, calc_weights
from segmentation.dataset.Dataset import DigitDataset
from segmentation.model.inst_loss import InstLoss
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from segmentation.utils.scheduler import GradualWarmupScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Trainer(object):

    def __init__(self, args):
        self.args = args
        # split train and valid dataset in each folder
        self.total_num = 25000

        self.model = DeepLab(backbone='resnet', output_stride=8, num_classes=11, freeze_bn=False)

        train_params = [{'params': self.model.get_backbone_params(), 'lr': 0.0001},
                        {'params': self.model.get_head_params(), 'lr': 0.0001},
                        {'params': self.model.get_inst_params(), 'lr': 0.0001 * 10}]
        # lr scheduler
        # wait to add
        self.optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=0.0005, nesterov=True)
        self.scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.3, verbose=True)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=10, total_epoch=3, after_scheduler=self.scheduler_plateau)
        weight = calc_weights(save_path='/app/Yibao-Cup-code/segmentation/dataset/class_weights.npy')
        weight = torch.Tensor(weight)
        # print(weight.shape)
        self.criterion = FocalLoss(weight=weight)
        self.criterion2 = InstLoss()

        # use cuda
        self.device_ids = [0, 1]
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model = self.model.cuda()

        self.best_val_loss = 100000

        # the unit is batch
        self.train_interval = 1000
        self.valid_interval = 2800

        self.epoch = 0
        self.batch = 0

        self._split_dataset()

        if args.inst is True:
            if args.resume is True:
                files = os.listdir('/app/Yibao-Cup-code/workspace/save2/snapshoot/')
                if len(files) == 0:
                    print('Have no  model file to resume')

                sorted_files = sorted(files,
                                      key=lambda x: os.path.getmtime('/app/Yibao-Cup-code/workspace/save2/snapshoot/' + x),
                                      reverse=True)
                file = sorted_files[0]
                resume = '/app/Yibao-Cup-code/workspace/save2/snapshoot/' + file
                self.epoch = int(file.split('_')[1])
                self.batch = int(file.split('_')[3])

                files = os.listdir('/app/Yibao-Cup-code/workspace/save2/best/')

                sorted_files = sorted(files,
                                      key=lambda x: os.path.getmtime('/app/Yibao-Cup-code/workspace/save2/best/' + x),
                                      reverse=True)
                file = sorted_files[0]
                self.best_val_loss = float(file.split('_')[5][:-4])

                print('Resume from epoch {}, batch {}, best val loss {}'.format(int(file.split('_')[1]),
                                                                                               int(file.split('_')[3]),
                                                                                               float(file.split('_')[5][
                                                                                                     :-4])))
                # need to train new added layers
                # so strict must be False
                self.model.load_state_dict(torch.load(resume), strict=False)
            else:
                files = os.listdir('/app/Yibao-Cup-code/workspace/save/best/')
                if len(files) == 0:
                    print('Have no pretrained model')

                sorted_files = sorted(files,
                                      key=lambda x: os.path.getmtime('/app/Yibao-Cup-code/workspace/save/best/' + x),
                                      reverse=True)
                file = sorted_files[0]
                resume = '/app/Yibao-Cup-code/workspace/save/best/' + file

                print('Load pretrained model from epoch {}, batch {}, best val loss {}'.format(int(file.split('_')[1]), int(file.split('_')[3]), float(file.split('_')[5][:-4])))
                # need to train new added layers
                # so strict must be False
                self.model.load_state_dict(torch.load(resume), strict=False)

        elif args.resume is True:
            files = os.listdir('/app/Yibao-Cup-code/workspace/save/snapshoot/')
            if len(files) == 0:
                print('Have no model file to resume')

            sorted_files = sorted(files,
                                  key=lambda x: os.path.getmtime('/app/Yibao-Cup-code/workspace/save/snapshoot/' + x),
                                  reverse=True)
            file = sorted_files[0]
            resume = '/app/Yibao-Cup-code/workspace/save/snapshoot/' + file
            self.epoch = int(file.split('_')[1])
            self.batch = int(file.split('_')[3])

            files = os.listdir('/app/Yibao-Cup-code/workspace/save/best/')

            sorted_files = sorted(files, key=lambda x: os.path.getmtime('/app/Yibao-Cup-code/workspace/save/best/' + x),
                                  reverse=True)
            file = sorted_files[0]
            self.best_val_loss = float(file.split('_')[5][:-4])

            print('Resume from epoch {}, batch {}, best val loss {}'.format(self.epoch, self.batch, self.best_val_loss))
            self.model.load_state_dict(torch.load(resume))

    def _split_dataset(self):
        if os.path.exists('/app/Yibao-Cup-code/segmentation/trainval.npy'):
            serial_nums = np.load('/app/Yibao-Cup-code/segmentation/trainval.npy')
            train_serial_nums = serial_nums[0]
            valid_serial_nums = serial_nums[1]
        else:
            serial_nums = np.arange(self.total_num)
            train_serial_nums, valid_serial_nums = train_test_split(serial_nums, test_size=0.1, random_state=42)
            np.save('/app/Yibao-Cup-code/segmentation/trainval.npy', [train_serial_nums, valid_serial_nums])

        train_dataset1 = DigitDataset('/app/SegDataset/number_segment_1/img',
                                      '/app/SegDataset/number_segment_1/label',
                                      '/app/SegDataset/number_segment_1/label.txt',
                                      offset=0, ids=train_serial_nums)
        train_dataset2 = DigitDataset('/app/SegDataset/number_segment_2/img',
                                      '/app/SegDataset/number_segment_2/label',
                                      '/app/SegDataset/number_segment_2/label.txt',
                                      offset=25000, ids=train_serial_nums + 25000)
        valid_dataset1 = DigitDataset('/app/SegDataset/number_segment_1/img',
                                      '/app/SegDataset/number_segment_1/label',
                                      '/app/SegDataset/number_segment_1/label.txt',
                                      offset=0, ids=valid_serial_nums)
        valid_dataset2 = DigitDataset('/app/SegDataset/number_segment_2/img',
                                      '/app/SegDataset/number_segment_2/label',
                                      '/app/SegDataset/number_segment_2/label.txt',
                                      offset=25000, ids=valid_serial_nums + 25000)
        self.train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        self.valid_dataset = ConcatDataset([valid_dataset1, valid_dataset2])
        # batch size should be 4 * gpu for TITAN Xp
        self.batch_size = 4 * len(self.device_ids)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8,
                                       pin_memory=True, drop_last=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=8,
                                       pin_memory=True, drop_last=True)

    def validating(self, limit=160):
        valid_loss = 0.0
        self.model.eval()
        valid_num = len(self.valid_loader)
        print('Start validation')
        # t = tqdm(self.valid_loader)
        # no need for backward
        # there's a bug I cannot figure out
        # if with grad, out of memory will occur
        # however, in training process, this situation will not occur
        with torch.no_grad():
            for i, sample in enumerate(self.valid_loader):
                if i >= limit:
                    break
                torch.cuda.empty_cache()
                img, mask_multi_hot, _, cls_num, loc = sample
                img, mask_multi_hot, cls_num, loc = img.cuda(), mask_multi_hot.cuda(), cls_num.cuda(), loc.cuda()
                output, inst_num, inst_loc = self.model(img)
                loss = None
                for j in range(self.batch_size):
                    output_t = output[j].reshape(11, -1).permute(1, 0)
                    mask_multi_hot_t = mask_multi_hot[j].reshape(11, -1).permute(1, 0)
                    if loss is None:
                        loss = self.criterion(output_t, mask_multi_hot_t).item()
                    else:
                        loss += self.criterion(output_t, mask_multi_hot_t).item()

                output = torch.sigmoid(output)
                t1 = torch.tensor([1]).cuda()
                t0 = torch.tensor([0]).cuda()
                output = torch.where(output > 0.5, t1, t0)
                weight = torch.where(output[:, 10, :, :] == 1, t0, t1).reshape(
                    (output.shape[0], 1, output.shape[2], output.shape[3]))
                inst_loss = self.criterion2(inst_num, cls_num, inst_loc, loc, weight).item()

                valid_loss += (loss + inst_loss * self.batch_size) / self.batch_size

            if limit < len(self.valid_loader):
                valid_loss = valid_loss / limit
            else:
                valid_loss = valid_loss / len(self.valid_loader)

        # t.close()

        return valid_loss

    def training(self, epoch):
        if epoch < self.epoch:
            return
        train_loss = 0.0
        self.model.train()
        # train_num = len(self.train_loader)
        print('Start epoch {}'.format(epoch))
        # t = tqdm(self.train_loader)
        for i, sample in enumerate(self.train_loader):
            # if i < self.batch:
            #     # print(i)
            #     continue
            torch.cuda.empty_cache()
            # t = time.time()
            img, mask_multi_hot, _, cls_num, loc = sample
            # print('Load data for {}s'.format(time.time() - t))
            # print(mask_multi_hot.shape)
            img, mask_multi_hot, cls_num, loc = img.cuda(), mask_multi_hot.cuda(), cls_num.cuda(), loc.cuda()
            self.optimizer.zero_grad()
            # t = time.time()
            # we can get (4, 11, 400, 400)
            output, inst_num, inst_loc = self.model(img)
            # print('Forward input for {}s'.format(time.time() - t))
            # print(output.shape)
            # print(mask_multi_hot.shape)
            # output = output.view(4, 11, -1)
            # mask_multi_hot = mask_multi_hot.view(4, 11, -1)
            # print(output.shape)
            # print(mask_multi_hot.shape)
            # t = time.time()

            loss = None
            for j in range(self.batch_size):
                output_t = output[j].reshape(11, -1).permute(1, 0)
                mask_multi_hot_t = mask_multi_hot[j].reshape(11, -1).permute(1, 0)
                # print(output_t.shape)
                # print(mask_multi_hot_t.shape)
                if loss is None:
                    loss = self.criterion(output_t, mask_multi_hot_t)
                else:
                    loss += self.criterion(output_t, mask_multi_hot_t)
            # loss = self.criterion(output, mask_multi_hot)
            # print('Calc loss for {}s'.format(time.time() - t))
            # t = time.time()

            # need to modify smooth_l1_loss
            # (4, 11, 400, 400)
            output = torch.sigmoid(output)
            t1 = torch.tensor([1]).cuda()
            t0 = torch.tensor([0]).cuda()
            output = torch.where(output > 0.5, t1, t0)
            weight = torch.where(output[:, 10, :, :] == 1, t0, t1).reshape(
                (output.shape[0], 1, output.shape[2], output.shape[3]))
            inst_loss = self.criterion2(inst_num, cls_num, inst_loc, loc, weight)

            # print(inst_loss)
            loss += inst_loss * self.batch_size
            loss.backward()
            # print('Backward for {}s'.format(time.time() - t))
            # t = time.time()
            self.optimizer.step()
            # print('Update weights for {}s'.format(time.time() - t))
            train_loss += loss.item() / self.batch_size

            if i == 0:
                continue

            if i % self.train_interval == 0:
                train_loss = train_loss / self.train_interval
                print('Epoch:{}, batch:{}, train loss:{}'.format(epoch, i, train_loss))
                train_loss = 0.0

            if i % self.valid_interval == 0:
                del img, mask_multi_hot, cls_num, loc, output, inst_num, inst_loc, weight, loss
                valid_loss = self.validating()
                self.model.train()
                self.scheduler.step(valid_loss, 2*epoch + int(i/self.valid_interval))
                print('Epoch:{}, batch:{}, valid loss:{}'.format(epoch, i, valid_loss))
                if valid_loss < self.best_val_loss and epoch > 1:
                    self.best_val_loss = valid_loss
                    # save model
                    # parameters are module.xxx.xxx
                    # if test is not on multi gpu
                    # need to transform state dict
                    torch.save(self.model.state_dict(),
                               '/app/Yibao-Cup-code/workspace/save2/best/epoch_{}_batch_{}_valloss_{}.pth'.format(epoch,
                                                                                                                 i,
                                                                                                                 self.best_val_loss))
                torch.save(self.model.state_dict(),
                           '/app/Yibao-Cup-code/workspace/save2/snapshoot/epoch_{}_batch_{}_valloss_{}.pth'.format(epoch,
                                                                                                                  i,
                                                                                                                  valid_loss))
        self.batch = 0
        self._split_dataset()
        # t.close()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser(description='Pytorch deeplabv3+ training')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from the newest model')
    parser.add_argument('--inst', action='store_true', default=False, help='train instance-level net')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(args)
    for epoch in range(0, 60):
        trainer.training(epoch)
