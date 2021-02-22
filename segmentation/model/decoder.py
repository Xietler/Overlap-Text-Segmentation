# -*- coding: utf-8 -*-
# @Time    : 2019/2/23 13:41
# @Author  : Ruichen Shao
# @File    : decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU(inplace=True)
        # concat low level feature and x
        # so the input dim = 256 + 48 = 304
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        output = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        output = torch.cat((output, low_level_feat), dim=1)
        output = self.last_conv(output)

        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    model = Decoder(3, 'resnet', nn.BatchNorm2d)
    x = torch.rand(2, 256, 64, 64)
    low_level_feat = torch.rand(2, 256, 512, 512)
    output = model(x, low_level_feat)
    print(output.size())