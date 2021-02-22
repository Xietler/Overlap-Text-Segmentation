# -*- coding: utf-8 -*-
# @Time    : 2019/2/28 20:57
# @Author  : Ruichen Shao
# @File    : test.py
import sys

sys.path.insert(0, '/app/Yibao-Cup-code')
from segmentation.utils.transformer import Transformer
from segmentation.model.deeplab import DeepLab
from segmentation.utils.metrics import Evaluator
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
# import h5py


def build_model(model_path, use_cuda=False):
    transformer = Transformer(11)
    model = DeepLab(backbone='resnet', output_stride=8, num_classes=11, freeze_bn=False, use_cuda=use_cuda)
    if use_cuda is True:
        device_ids = [0, 1]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        model.load_state_dict(transformer.module2model_dict(model_path), strict=False)
    model.eval()
    return model

def forward_on_batch(dir_path, offset, batch_size, batch_num, model):
    # dataset = DigitDataset(dir_path + '/img',
    #                        dir_path + '/label',
    #                        dir_path + '/label.txt',
    #                        offset=offset)
    dataset = DigitDataset(dir_path + '/img',
                          dir_path + '/label',
                          None,
                          offset=offset)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True)
    ret1 = None
    ret2 = None
    ret3 = None
    for i, sample in enumerate(dataloader):
        if i == batch_num:
            break
        # print(os.path.exists('/app/Yibao-Cup-code/cache2/batch_{}.h5'.format(int(i+offset/4))))
        if os.path.exists('/app/Yibao-Cup-code/cache2/batch_{}.h5'.format(int(i+offset/4))) is not True:
            imgs, mask_multi_hots, _, _, _ = sample
            # imgs = sample
            outputs, inst_nums, inst_locs = model(imgs)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.detach().cpu().numpy()
            inst_nums = inst_nums.detach().cpu().numpy()
            inst_locs = inst_locs.detach().cpu().numpy()
            outputs = np.asarray(outputs, dtype=np.float16)
            inst_nums = np.asarray(inst_nums, dtype=np.float16)
            inst_locs = np.asarray(inst_locs, dtype=np.float16)
            f = h5py.File('/app/Yibao-Cup-code/cache2/batch_{}.h5'.format(int(i+offset/4)), 'w')
            f.create_dataset('outputs', data=outputs, compression='gzip')
            f.create_dataset('inst_nums', data=inst_nums, compression='gzip')
            f.create_dataset('inst_locs', data=inst_locs, compression='gzip')
            f.close()
            # np.savez('batch_{}.npz'.format(int(i+offset/4)), outputs=outputs, inst_nums=inst_nums, inst_locs=inst_locs)
            print('Save batch_{} outputs.'.format(int(i+offset/4)))

        f = h5py.File('/app/Yibao-Cup-code/cache2/batch_{}.h5'.format(int(i+offset/4)), 'r')
        # npz = np.load('batch_{}.npz'.format(i))
        if ret1 is None:
            ret1 = f['outputs']
            ret2 = f['inst_nums']
            ret3 = f['inst_locs']
            # ret1 = npz['outputs']
            # ret2 = npz['inst_nums']
            # ret3 = npz['inst_locs']
        else:
            ret1 = np.concatenate((ret1, f['outputs']), axis=0)
            ret2 = np.concatenate((ret2, f['inst_nums']), axis=0)
            ret3 = np.concatenate((ret3, f['inst_locs']), axis=0)
    return ret1, ret2, ret3

def adjust_threshold(outputs, inst_nums, inst_locs,offset, dir_path):
    transformer = Transformer(11)
    tmp = outputs
    max_maps = np.zeros((tmp.shape[0],))
    best_thresholds = np.zeros((tmp.shape[0],))
    kernel = np.ones((3, 3), np.uint8)
    output = np.where(tmp > 0.53, 1, 0)
    for idx in range(offset, offset+tmp.shape[0]):
        # if os.path.exists('/app/Yibao-Cup-code/result/label_{}.png'.format(idx)) is True:
        #     continue
        # if idx != 25014:
        #     continue
        img = Image.open(dir_path+'/img/img_{}.jpg'.format(idx))
        idx_h, idx_w = np.where(output[idx-offset, :-1].sum(axis=0) >= 1)
        # mask = Image.open(dir_path+'/label/label_{}.png'.format(idx))
        # mask = np.array(mask, dtype=np.uint8)
        # idx_h, idx_w = np.where(mask != 0)
        # mask_ = mask[idx_h, idx_w]
        # total = (mask_ == 60).sum() + (mask_ == 120).sum() + 8 * (mask_ == 180).sum()
        # for k in range(53, 54):
        #     threshold = k / 100
        #     output = np.where(tmp > threshold, 1, 0)
            # print(output.shape)
            # img = transformer.multi_hot2rgb(output[idx-25000])
        mask_1, mask_2, mask_3, mask_4, flag = transformer.multi_hot2mask(output[idx-offset], inst_nums[idx-offset], np.concatenate((np.asarray(img).transpose((2, 0, 1)),), axis=0))

        if flag is True:
            mask_3 = cv2.morphologyEx(mask_3, cv2.MORPH_OPEN, kernel)
            mask_3 = cv2.morphologyEx(mask_3, cv2.MORPH_CLOSE, kernel)
            mask_4 = cv2.morphologyEx(mask_4, cv2.MORPH_OPEN, kernel)
            mask_4 = cv2.morphologyEx(mask_4, cv2.MORPH_CLOSE, kernel)
            mask_3_ = np.where(mask_3 >= 128, 60, 90)
            mask_4_ = np.where(mask_4 >= 128, 120, 90)
            mask_1 = mask_3_ + mask_4_
            mask_1 = np.where(mask_1 == 150, 60, mask_1)
            mask_1 = np.where(mask_1 == 210, 120, mask_1)
            mask_3_ = np.where(mask_3 >= 128, 120, 90)
            mask_4_ = np.where(mask_4 >= 128, 60, 90)
            mask_2 = mask_3_ + mask_4_
            mask_2 = np.where(mask_2 == 150, 60, mask_2)
            mask_2 = np.where(mask_2 == 210, 120, mask_2)
            t = 0
            for i in range(400):
                for j in range(400):
                    if t < len(idx_h) and idx_h[t] == i and idx_w[t] == j:
                        t += 1
                    else:
                        mask_1[i, j] = 0
        # print(mask_1.shape)
        # mask_1 = mask_1[idx_h, idx_w]
        # mask_2 = mask_2[idx_h, idx_w]
        # print(mask_1.shape)

            # intersection_1 = np.logical_and(mask_1 == mask_, mask_1 == 60).sum() + np.logical_and(mask_1 == mask_, mask_1 == 120).sum() + 8 * np.logical_and(mask_1 == mask_, mask_1 == 180).sum()
            # intersection_2 = np.logical_and(mask_2 == mask_, mask_2 == 60).sum() + np.logical_and(mask_2 == mask_, mask_2 == 120).sum() + 8 * np.logical_and(mask_2 == mask_, mask_2 == 180).sum()
            # print(mask_1.shape)
            # print(mask.shape)
            # print(mask_1 == mask)
            # print(mask_1 == 60)
            # print(np.logical_and(mask_1 == mask, mask_1 == 60).sum())
            # iou = max(intersection_1 / total, intersection_2 / total)
        try:
            Image.fromarray(np.asarray(mask_1, dtype=np.uint8)).save('/app/Yibao-Cup-code/result/label_{}.png'.format(idx))
        except:
            print(idx)
            # print('adjust map for img_{}, threshold_{} is {}'.format(idx, threshold, iou))
            # if iou > max_maps[idx-25000]:
            #     max_maps[idx-25000] = iou
            #     best_thresholds[idx-25000] = threshold
    # best thresholds are 0.37, 0.49, 0.52, 0.51,
    #                     0.57, 0.41, 0.52, 0.76,
    #                     0.52, 0.46, 0.48, 0.60,
    #                     0.49, 0.65, 0.64, 0.59,
    #                     0.52, 0.49, 0.50, 0.47,
    # so maybe 0.50~0.55 is a good trade-off
    # for i in range(tmp.shape[0]):
    #     print('img {}\'s max map is {}, best threshold is {}'.format(i+25000, max_maps[i], best_thresholds[i]))

if __name__ == '__main__':
    from segmentation.dataset.Dataset import DigitDataset
    from torch.utils.data.dataloader import DataLoader
    from PIL import Image

    model_path = 'C:/Users/SRC/PycharmProjects/DigitSegmentation/final2/epoch_42_batch_2800_valloss_0.26054752129393527.pth'
    dir_paths = ['C:/Users/SRC/Downloads/SegDataset/number_segment_1', 'C:/Users/SRC/Downloads/SegDataset/number_segment_2']
    # model_path = '/app/Yibao-Cup-code/final2/epoch_41_batch_5600_valloss_0.26054752129393527.pth'
    # dir_paths = ['/app/SegDataset/number_segment_1', '/app/SegDataset/number_segment_2', '/app/SegDataset/test_dataset']
    model = build_model(model_path, use_cuda=False)
    # outputs, inst_nums, inst_locs = forward_on_batch(dir_paths[1], 25000, 4, 5, model)
    # outputs, inst_nums, inst_locs = forward_on_batch(dir_paths[2], 0, 4, 250, model)

    img = np.asarray(Image.open('123.jpg')).transpose((2, 0, 1))
    print(img.shape)
    output, inst_num, inst_loc = model(torch.FloatTensor([img]))
    output = output.detach().numpy()
    output = np.where(output > 0.53, 1, 0)
    transformer = Transformer(11)
    img = transformer.multi_hot2rgb(output[0])
    Image.fromarray(img).show()
    # evaluator = Evaluator(11)
    # acc = []
    # for batch in range(12500):
    #     npz = np.load('/app/Yibao-Cup-code/cache/batch_{}.npz'.format(batch))
    #     outputs = npz['outputs']
    #     inst_nums = npz['inst_nums']
    #     inst_locs = npz['inst_locs']
    #     outputs = np.where(outputs > 0.53, 1, 0)
    #     for i in range(4):
    #         if batch < 6250:
    #             img = Image.open('/app/SegDataset/number_segment_1/img/img_{}.jpg'.format(batch*4+i))
    #             mask = Image.open('/app/SegDataset/number_segment_1/label/label_{}.png'.format(batch*4+i))
    #         else:
    #             img = Image.open('/app/SegDataset/number_segment_2/img/img_{}.jpg'.format(batch*4+i))
    #             mask = Image.open('/app/SegDataset/number_segment_2/label/label_{}.png'.format(batch * 4 + i))
    #         # print(inst_nums[i])
    #         mask_1, mask_2, mask_3, mask_4 = transformer.multi_hot2mask(outputs[i], inst_nums[i], np.asarray(img).transpose((2, 0, 1)))
    #         iou = evaluator.calc_valid_iou(mask, mask_1, mask_2)
    #         acc.append(iou)
    #         print('img_{}\'s acc is {}'.format(batch*4+i, iou))
    # acc = np.asarray(acc)
    # print('avg acc is {}'.format(acc.mean()))
    # outputs = np.where(outputs > 0.53, 1, 0)

    # adjust_threshold(outputs, inst_nums, inst_locs, 0, dir_paths[2])

    # output = np.where(output > 0.53, 1, 0)
    # img = transformer.multi_hot2rgb(output)
    # tmp = output.reshape(11, -1)
    # print(tmp.sum(axis=1))
    # img = Image.fromarray(img)
    # img.show()
