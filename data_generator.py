# -*- coding: utf-8 -*-
# @Time    : 2019/1/16 16:11
# @Author  : Ruichen Shao
# @File    : data_generator.py

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random, string
from config import Config

class DataGenerator(object):
    def __init__(self):
        self.width = 400
        self.height = 400
        self.cfg = Config()

    # 获取随机颜色
    def getRandomColor(self):
        return (random.randint(30, 225), random.randint(30, 225), random.randint(30, 225))

    # 获取随机字符(a-z, A-Z, 0-9)
    def getRandomChar(self):
        return random.choice(string.ascii_letters + string.digits)

    def chToIdx(self, ch):
        if ch >= '0' and ch <= '9':
            return ord(ch) - ord('0')
        if ch >= 'a' and ch <= 'z':
            return ord(ch) - ord('a') + 10
        if ch >= 'A' and ch <= 'Z':
            return ord(ch) - ord('A') + 10 + 26

    # 绘制单张图片
    def getPicture(self, id):
        # 创建画布
        image = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        self.font = ImageFont.truetype("arial.ttf", size=random.randint(50, 150))
        draw = ImageDraw.Draw(image)
        ch = self.getRandomChar()
        # ch = '0'
        draw.text((random.randint(50, 250), random.randint(50, 250)), ch, font=self.font, fill=self.getRandomColor())
        image.save(self.cfg.workspace + '/' + self.mode + '/{0:06d}'.format(id) + '.png')
        self.labels.append(self.chToIdx(ch))
        # print(self.chToIdx(ch))
        # image.show()

    # 生成数据集
    def genDataset(self, mode='train'):
        self.mode = mode
        self.labels = []
        if mode == 'train':
            num = 10000
        elif mode == 'val':
            num = 2000
        elif mode == 'test':
            num = 5000
        elif mode == 'debug':
            num = 1
        print('start generate ' + mode + ' dataset')
        for i in range(num):
            self.getPicture(i + 1)
            if i % 100 == 0:
                print(i / num)
        t = np.asarray(self.labels)
        print(t)
        np.save(self.cfg.workspace + '/' + mode + '.npy', t)
        print('complete generate ' + mode + ' dataset')

if __name__ == '__main__':
    dg = DataGenerator()
    # dg.genDataset('debug')
    dg.genDataset('train')
    dg.genDataset('val')
    dg.genDataset('test')
