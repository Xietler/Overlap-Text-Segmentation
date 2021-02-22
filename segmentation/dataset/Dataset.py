# -*- coding: utf-8 -*-
# @Time    : 2019/2/22 14:06
# @Author  : Ruichen Shao
# @File    : Dataset.py

import torch
import os
import numpy as np
import cv2
from torch.utils import data

class DigitDataset(data.Dataset):

    def __init__(self, im_dir, mask_dir, label_path=None, postfix='.png', offset=0, ids=None, num_classes=10):
        super().__init__()
        self.im_dir = im_dir
        self.mask_dir = mask_dir
        self.label_path = label_path
        self.postfix = postfix
        self.num_classes = num_classes
        self.mode = 'test'
        if ids is None:
            self.len = len(os.listdir(im_dir))
            self.ids = np.arange(offset, self.len+offset)
        else:
            self.len = len(ids)
            self.ids = ids
        self.files = self._txt2label()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data_file = self.files[index]
        # shape is (H, W, C), channel order is BGR
        img = cv2.imread(data_file['img'], cv2.IMREAD_COLOR)
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # baseline doesn't distinguish 60 and 120
        # here we take 0-9 as 10 classes
        # label 10 represents background
        if self.mode == 'test':
            img = np.asarray(img, np.float32)
            return img.transpose((2, 0, 1))
        else:
            mask = cv2.imread(data_file['mask'], cv2.IMREAD_GRAYSCALE)
            # print(len(np.where(mask==60)[0]))
            # print(np.where(mask == 120))
            label1 = data_file['label1']
            label2 = data_file['label2']
            cls_num = np.zeros((self.num_classes,))
            cls_num[label1] += 1
            cls_num[label2] += 1
            id = data_file['id']
            img = np.asarray(img, np.float32)
            mask = np.asarray(mask, np.int32)
            loc = np.zeros((6, 400, 400), dtype=np.float)
            idx_y, idx_x = np.where(np.logical_or(mask==60, mask==180))
            lx = np.min(idx_x)
            ly = np.min(idx_y)
            rx = np.max(idx_x)
            ry = np.max(idx_y)
            cx = (lx + rx) / 2
            cy = (ly + ry) / 2
            # img is 400 * 400
            loc1 = np.array([cx/400, cy/400, lx/400, ly/400, rx/400, ry/400], dtype=np.float)
            # print(loc1)
            idx_y, idx_x = np.where(np.logical_or(mask==120, mask==180))
            lx = np.min(idx_x)
            ly = np.min(idx_y)
            rx = np.max(idx_x)
            ry = np.max(idx_y)
            cx = (lx + rx) / 2
            cy = (ly + ry) / 2
            # img is 400 * 400
            loc2 = np.array([cx / 400, cy / 400, lx / 400, ly / 400, rx / 400, ry / 400], dtype=np.float)
            # print(loc2)
            idx_y, idx_x = np.where(mask==60)
            for i in range(len(idx_y)):
                loc[:, idx_y[i], idx_x[i]] = loc1
            idx_y, idx_x = np.where(mask == 120)
            for i in range(len(idx_y)):
                loc[:, idx_y[i], idx_x[i]] = loc2
            idx_y, idx_x = np.where(mask == 180)
            for i in range(len(idx_y)):
                loc[:, idx_y[i], idx_x[i]] = (loc1 + loc2) / 2
            # print(mask.shape)
            # background
            mask = np.where(mask==0, 10, mask)
            mask = np.where(mask==60, label1, mask)
            mask = np.where(mask==120, label2, mask)
            # this order is incorrect
            # because 0 maybe label
            # mask = np.where(mask == 0, 10, mask)
            mask_1 = np.where(mask==180, label1, mask).reshape(-1)
            mask_2 = np.where(mask==180, label2, mask).reshape(-1)
            # get multi hot format
            mask_multi_hot = self._make_multi_hot(mask_1, mask_2, self.num_classes+1).reshape((mask.shape[0], mask.shape[1], 11), order='C')
            # print(mask_multi_hot.shape)
            # print(mask.shape)
            # transform from (H, W, C) to (C, H, W)
            img = img.transpose((2, 0, 1))
            mask_multi_hot = mask_multi_hot.transpose((2, 0, 1))
            # overlap = len(np.where(mask==180)[0])
            # mask = np.where(mask==180, label1, mask).reshape(-1)
            # mask = np.concatenate((mask, [label2] * overlap))
            # mask = np.asarray(mask, np.int32)
            # print(mask_multi_hot.shape)
            # print(img.dtype, mask_multi_hot.dtype, mask.dtype)

            # delete return mask to make sure the dims are the same
            return img, mask_multi_hot, id, cls_num, loc

    def _make_multi_hot(self, mask_1, mask_2, class_num):
        # if need column vector
        # add transpose
        mask_1 = np.eye(class_num)[mask_1]
        mask_2 = np.eye(class_num)[mask_2]
        mask = mask_1 + mask_2
        mask = np.where(mask == 2, 1, mask)
        # print(mask)
        return mask

    def _txt2label(self):
        files = []
        if self.label_path is None:
            for id in self.ids:
                im_file = os.path.join(self.im_dir, 'img_' + str(id) + '.jpg')
                mask_file = os.path.join(self.mask_dir, 'label_' + str(id) + self.postfix)
                files.append({
                    'img': im_file,
                    'mask': mask_file,
                    'id': id,
                    'label1': None,
                    'label2': None,
                })
        else:
            with open(self.label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    t = line.split(',')
                    img = t[0]
                    im_file = os.path.join(self.im_dir, img)
                    id = img[4:-4]
                    if int(id) not in self.ids:
                        continue
                    mask_file = os.path.join(self.mask_dir, 'label_' + id + self.postfix)
                    # label 60
                    label1 = t[1][6:7]
                    # label 120
                    label2 = t[2][7:8]
                    # print(img)
                    # print(id)
                    # print(label1, label2)
                    files.append({
                        'img': im_file,
                        'mask': mask_file,
                        'id': id,
                        'label1': int(label1),
                        'label2': int(label2),
                    })
        return files

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from PIL import Image
    dataset = DigitDataset('C:/Users/SRC/Downloads/SegDataset/number_segment_2/img',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_2/label',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_2/label.txt',
                           offset=25000)
    from segmentation.cluster.fcm import cmeans
    img, mask_multi_hot, id, cls_num, loc = dataset[14]
    idx_h, idx_w = np.where(mask_multi_hot[6] == 1)
    data = loc[:, idx_h, idx_w]
    N = len(idx_h)
    # print(np.max(inst_loc))
    # print(inst_loc)
    # N * 6
    data = loc[:, idx_h, idx_w]
    data = data.T
    print(data)
    # N * 2
    loc_xy = np.array([xy for xy in zip(idx_h, idx_w)])
    # print(data)
    data = torch.tensor(data, dtype=torch.float32)
    v, _, u, _, _, _ = cmeans(data, 2, 2, 0.001, 1000, torch.cat(
        (data[0].reshape((1, data.shape[1])), data[-1].reshape(1, data.shape[1]))))
    print(v)
    threshold = 0.4
    # mask only on foreground pixels
    # consider others to be bg
    mask_1 = np.zeros((400, 400), dtype=np.int8)
    mask_2 = np.zeros((400, 400), dtype=np.int8)
    # only one digit
    mask_3 = np.zeros((400, 400), dtype=np.int8)
    mask_4 = np.zeros((400, 400), dtype=np.int8)
    for i in range(N):
        # if (v.sum() / 2 <= data[i, :].sum()):
        if u[0, i] <= threshold:
            mask_1[loc_xy[i][0], loc_xy[i][1]] = 60
            mask_2[loc_xy[i][0], loc_xy[i][1]] = 120
            mask_3[loc_xy[i][0], loc_xy[i][1]] = 180
        # elif (v.sum() / 2 > data[i, :].sum()):
        elif u[0, i] >= 1 - threshold:
            mask_1[loc_xy[i][0], loc_xy[i][1]] = 120
            mask_2[loc_xy[i][0], loc_xy[i][1]] = 60
            mask_4[loc_xy[i][0], loc_xy[i][1]] = 180
        else:
            # print(2)
            mask_1[loc_xy[i][0], loc_xy[i][1]] = mask_2[loc_xy[i][0], loc_xy[i][1]] = mask_3[
                loc_xy[i][0], loc_xy[i][1]] = mask_4[loc_xy[i][0], loc_xy[i][1]] = 180
    fig, axes = plt.subplots(4, 6, figsize=(10, 10))
    for i in range(1):
        # print(inst_nums[i])
        for j in range(6):
            if j == 0:
                pass
            elif j == 1:
                pass
            elif j == 2:
                tmp = Image.fromarray(mask_3)
                axes[i, j].imshow(tmp)
            elif j == 3:
                tmp = Image.fromarray(mask_4)
                axes[i, j].imshow(tmp)
            elif j == 4:
                tmp = Image.fromarray(mask_1)
                axes[i, j].imshow(tmp)
            elif j == 5:
                tmp = Image.fromarray(mask_2)
                axes[i, j].imshow(tmp)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.show()
    # b = dataset[14]


