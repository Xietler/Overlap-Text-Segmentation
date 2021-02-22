# -*- coding: utf-8 -*-
# @Time    : 2019/2/23 14:09
# @Author  : Ruichen Shao
# @File    : deeplab.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from segmentation.model.backbone.resnet import ResNet, Bottleneck
from segmentation.model.aspp import ASPP
from segmentation.model.decoder import Decoder
import math

class _InstanceConv(nn.Module):

    def __init__(self, inplanes, planes, BatchNorm, use_cuda=True):
        super(_InstanceConv, self).__init__()
        self.use_cuda = use_cuda
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        # add (x, y)
        self.conv2 = nn.Conv2d(inplanes+2, planes, kernel_size=1,
                                      stride=1, padding=0, bias=False)
        self.bn = BatchNorm(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        # if test on cpu
        # need to delete cuda
        if self.use_cuda:
            coordinate_map = torch.stack(
                torch.meshgrid((torch.arange(0, x.shape[2]), torch.arange(0, x.shape[3])))).float().cuda()
        else:
            coordinate_map = torch.stack(
                torch.meshgrid((torch.arange(0, x.shape[2]), torch.arange(0, x.shape[3])))).float()
        tup = ()
        for i in range(x.shape[0]):
            tup += (coordinate_map,)
        coordinate_map = torch.stack(tup)
        output = self.conv1(x)
        output = self.bn(output)
        output = self.relu(output)
        output = torch.cat((x, coordinate_map), dim=1)
        output = self.conv2(output)

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLab(nn.Module):

    def __init__(self, backbone='resnet', output_stride=8, num_classes=3, freeze_bn=False, resume=None, use_cuda=True):
        super(DeepLab, self).__init__()
        self.use_cuda = use_cuda
        BatchNorm = nn.BatchNorm2d

        # if you don't want to resume
        # please set pretrain = True
        self.backbone = ResNet(Bottleneck, 8, BatchNorm, 101, False)
        self.aspp = ASPP(backbone, 8, BatchNorm)
        self.decoder = Decoder(num_classes, backbone, BatchNorm)

        # add new layers for instance-level segmentation
        # this config is just for resnet
        # if you change backbone, maybe you need to change the config
        self.instcon1 = _InstanceConv(256, 48, BatchNorm, use_cuda)
        self.instcon2 = _InstanceConv(512, 96, BatchNorm, use_cuda)
        self.instcon3 = _InstanceConv(1024, 128, BatchNorm, use_cuda)
        self.instcon4 = _InstanceConv(256, 48, BatchNorm, use_cuda)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.instcon5 = nn.Conv2d(272, 48, kernel_size=1, bias=False)
        self.instcon6 = nn.Conv2d(48, 6, kernel_size=1, bias=False)
        self.instcon7 = nn.Conv2d(304, 32, kernel_size=2, stride=2, bias=False)
        # for instance segmentation we need to exclude background
        self.instcon8 = nn.Conv2d(32, num_classes-1, kernel_size=25, bias=False)
        self.instcon9 = nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=False)
        self.instcon10 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)
        # maybe kernel size = 1 is not suitable
        # need to change
        self.instcon11 = nn.Conv2d(16, 6, kernel_size=5, padding=2, bias=False)
        self.bn1 = BatchNorm(272)
        self.bn2 = BatchNorm(32)
        self.bn3 = BatchNorm(16)
        self.bn4 = BatchNorm(6)
        self.relu = nn.ReLU(inplace=True)

        if freeze_bn:
            self.freeze_bn()
        if resume is not None:
            self._load_resumed_model(resume)


    def forward(self, img):
        # t = time.time()
        x, low_level_feat1, low_level_feat2, low_level_feat3 = self.backbone(img)
        # (2048, 50, 50)
        # (256, 100, 100)
        # (512, 50, 50)
        # (1024, 50, 50)
        # print(x.shape, low_level_feat1.shape, low_level_feat2.shape, low_level_feat3.shape)
        # print('Backbone forward for {}s'.format(time.time() - t))
        # t = time.time()
        # 48 * 100 * 100
        low_level_feat1_inst = self.instcon1(low_level_feat1)
        # low_level_feat1_inst = self.downsample(low_level_feat1_inst)
        # 96 * 50 * 50
        low_level_feat2_inst = self.instcon2(low_level_feat2)
        # 128 * 50 * 50
        low_level_feat3_inst = self.instcon3(low_level_feat3)
        x = self.aspp(x)
        # 48 * 50 * 50
        low_level_feat4_inst = self.instcon4(x)
        # 272 * 50 * 50
        fusion_feat_inst = torch.cat((low_level_feat2_inst, low_level_feat3_inst, low_level_feat4_inst), dim=1)
        fusion_feat_inst = self.bn1(fusion_feat_inst)
        fusion_feat_inst = self.relu(fusion_feat_inst)
        # (N, 48, 50, 50)
        inst_loc = self.instcon5(fusion_feat_inst)
        # 304 * 50 * 50
        inst_num = torch.cat((x, inst_loc), dim=1)
        # (N, 48, 100, 100)
        inst_loc = F.interpolate(inst_loc, size=[100, 100], mode='bilinear', align_corners=True)
        # 96 * 100 * 100
        inst_loc = torch.cat((low_level_feat1_inst, inst_loc), dim=1)
        # 32 * 100 * 100
        inst_loc = self.instcon9(inst_loc)
        inst_loc = self.bn2(inst_loc)
        inst_loc = self.relu(inst_loc)
        # 32 * 200 * 200
        inst_loc = F.interpolate(inst_loc, size=[200, 200], mode='bilinear', align_corners=True)
        # 16 * 200 * 200
        inst_loc = self.instcon10(inst_loc)
        inst_loc = self.bn3(inst_loc)
        inst_loc = self.relu(inst_loc)
        inst_loc = F.interpolate(inst_loc, size=img.size()[2:], mode='bilinear', align_corners=True)
        inst_loc = self.instcon11(inst_loc)
        inst_loc = self.bn4(inst_loc)
        inst_loc = self.relu(inst_loc)
        # print(inst_loc.shape)
        inst_num = self.instcon7(inst_num)
        inst_num = self.bn2(inst_num)
        inst_num = self.relu(inst_num)
        inst_num = self.instcon8(inst_num)
        inst_num = inst_num.squeeze()
        # (N, 11, 1, 1)
        # print(inst_num.shape)
        # (N, 256, 50, 50)
        # print(x.shape)
        # print('ASPP forward for {}s'.format(time.time() - t))
        # t = time.time()
        x = self.decoder(x, low_level_feat1)
        # before interpolate we need to add instance-level segmentation
        output = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)
        # print('Decoder forward for {}s'.format(time.time() - t))

        return output, inst_num, inst_loc

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_backbone_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_head_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    # return instance-level layers' parameters
    def get_inst_params(self):
        for m in self.children():
            if isinstance(m, _InstanceConv):
                for mm in m.named_modules():
                    if isinstance(mm[1], nn.Conv2d) or isinstance(mm[1], nn.BatchNorm2d):
                        for p in mm[1].parameters():
                            if p.requires_grad:
                                yield p

            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p

    def _load_resumed_model(self, path):
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        resume_dict = torch.load(path)
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in resume_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    model = DeepLab('resnet', 8, 3, False)
    # use batch normalization
    # so the batch size cannot be 1
    # model.get_inst_params()
    x = torch.rand(2, 3, 512, 512)
    output = model(x)
    print(output.size())


