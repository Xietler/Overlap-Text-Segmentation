# -*- coding: utf-8 -*-
# @Time    : 2019/3/1 10:37
# @Author  : Ruichen Shao
# @File    : visualize.py

import sys

sys.path.insert(0, 'C:/Users/SRC/PycharmProjects/DigitSegmentation')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PIL import Image
from segmentation.utils.transformer import Transformer
import cv2

if __name__ == '__main__':
    # img1 = Image.open('6_1.png')
    # kernel = np.ones((3, 3), np.uint8)
    # # img = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
    # img1 = cv2.morphologyEx(np.array(img1), cv2.MORPH_OPEN, kernel)
    # img1 = cv2.morphologyEx(np.array(img1), cv2.MORPH_CLOSE, kernel)
    # # img1 = Image.fromarray(img1)
    # img2 = Image.open('6_2.png')
    # kernel = np.ones((3, 3), np.uint8)
    # # img = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
    # img2 = cv2.morphologyEx(np.array(img2), cv2.MORPH_OPEN, kernel)
    # img2 = cv2.morphologyEx(np.array(img2), cv2.MORPH_CLOSE, kernel)
    # # img2 = Image.fromarray(img2)
    # img = img1 + img2
    # Image.fromarray(img).show()

    transformer = Transformer(11)
    fig, axes = plt.subplots(4, 6, figsize=(10, 10))
    imgs = []
    npz = np.load('batch_3.npz')
    threshold = 0.53
    outputs = npz['outputs']
    inst_nums = npz['inst_nums']
    inst_locs = npz['inst_locs']
    # print(inst_locs[0])
    # print(inst_locs.shape)
    # print(np.max(inst_locs))
    # print(np.min(inst_locs))
    # print(inst_locs[0,:,200,200])
    outputs = np.where(outputs > threshold, 1, 0)
    # print(inst_nums)

    for i in range(25012, 25016):
        imgs.append(Image.open('C:/Users/SRC/Downloads/SegDataset/number_segment_2/img/img_{}.jpg'.format(i)))

    for i in range(4):
        img = transformer.multi_hot2rgb(outputs[i])
        # print(inst_nums[i])
        mask_1, mask_2, mask_3, mask_4, _ = transformer.multi_hot2mask(outputs[i], inst_nums[i], np.concatenate((np.asarray(imgs[i]).transpose((2, 0, 1)),), axis=0))
        for j in range(6):
            if j == 0:
                axes[i, j].imshow(imgs[i])
            elif j == 1:
                tmp = Image.fromarray(img)
                axes[i, j].imshow(tmp)
            elif j == 2:
                tmp = Image.fromarray(mask_3, mode='L')
                axes[i, j].imshow(tmp, cmap='gray')
                # if i == 0:
                #     tmp.save('0_1.png')
            elif j == 3:
                tmp = Image.fromarray(mask_4, mode='L')
                axes[i, j].imshow(tmp, cmap='gray')
                # if i == 0:
                #     tmp.save('0_2.png')
            elif j == 4:
                tmp = Image.fromarray(mask_1, mode='L')
                axes[i, j].imshow(tmp, cmap='gray')
            elif j == 5:
                tmp = Image.fromarray(mask_2, mode='L')
                axes[i, j].imshow(tmp, cmap='gray')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.show()



