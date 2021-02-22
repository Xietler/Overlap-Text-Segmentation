# -*- coding: utf-8 -*-
# @Time    : 2019/2/28 18:56
# @Author  : Ruichen Shao
# @File    : transformer.py

import numpy as np
import torch
from PIL import Image
import random
import segmentation.cluster.fcm as FCM
from matplotlib import colors

cnames = {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    # 'black':                '#000000',
    'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen': '#90EE90',
    'lightgray': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'}


class Transformer(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.color_map = self._color_mapping()

    def _color_mapping(self):
        color_map = []
        index = 0
        colors = list(cnames.values())
        random.shuffle(colors)
        for c in colors:
            if index < self.num_classes - 1:
                color_map.append(self._hex2rgb(c))
            else:
                break
            index += 1
        # add black for background
        color_map.append(self._hex2rgb('#000000'))
        color_map = np.asarray(color_map)
        return color_map

    def _hex2rgb(self, str):
        str = str[1:]
        rgb = []
        for i in range(3):
            rgb.append(int(str[i * 2:i * 2 + 2], 16))
        rgb = np.asarray(rgb)
        return rgb

    # transform the output of model to rgb image
    # need sigmoid first
    def multi_hot2rgb(self, target):
        # target is (C, H, W)
        # target = target.numpy().transpose((1, 2, 0))
        target = target.transpose((1, 2, 0))
        # print(target.shape)
        img = np.zeros((target.shape[0], target.shape[1], 3))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                pass
                color = np.array([0, 0, 0])
                count = 0
                for index, label in enumerate(target[i, j, :]):
                    if label == 1:
                        color += self.color_map[index]
                        count += 1
                        continue
                if count != 0:
                    color = color // count
                img[i, j, :] = color
        return img.astype(dtype=np.uint8)

    # transform the output of model to mask for score
    def multi_hot2mask(self, target, inst_num, inst_loc):
        # target is (C, H, W)
        height = target.shape[1]
        width = target.shape[2]
        target = target.reshape(self.num_classes, -1)
        # target = target.transpose((1, 0))
        cls_sum = target.sum(axis=1)
        # print(cls_sum.shape)
        cls_idxs = np.argsort(-cls_sum)
        bg = cls_idxs[0]
        num1 = cls_idxs[1]
        num2 = cls_idxs[2]
        flag = False
        if cls_sum[num1] / cls_sum[num2] > 5:
            num2 = num1
            flag = True
        # print(cls_sum[num1], cls_sum[num2])
        target = target.reshape((self.num_classes, height, width))
        mask_1 = np.zeros((height, width), dtype=np.uint8)
        mask_2 = np.zeros((height, width), dtype=np.uint8)
        # only one digit
        mask_3 = np.zeros((height, width), dtype=np.uint8)
        mask_4 = np.zeros((height, width), dtype=np.uint8)
        # print(bg, num1, num2)
        # if num1 == num2
        # how to segment the instances
        # wait to solve
        if flag is True:
            # do cluster
            idx_h, idx_w = np.where(target[num1] == 1)
            N = len(idx_h)
            # print(np.max(inst_loc))
            # print(inst_loc)
            # N * 6
            # print(inst_loc.shape)
            # for i in range(400):
            #     for j in range(400):
            #         print(inst_loc[:, i, j])
            data = inst_loc[:, idx_h, idx_w]
            data = data.T
            # data = np.column_stack((data, idx_h / 100, idx_w / 100))
            # print(data.shape)
            # for i in range(data.shape[0]):
            #     print(data[i])
            # N * 2
            # data = inst_loc.reshape(6, -1)
            # data = data.T
            # idx_h = np.repeat(np.arange(400), 400)
            # idx_w = np.tile(np.arange(400), 400)
            loc_xy = np.array([xy for xy in zip(idx_h, idx_w)])
            # print(data)
            data = torch.tensor(data, dtype=torch.float32)
            # data -= 128
            v, _, u, _, _, _ = FCM.cmeans(data, 2, 2, 1e-6, 10000, torch.cat(
                (data[0].reshape((1, data.shape[1])), data[-1].reshape(1, data.shape[1]))))
            # _, _, u, _, _, _ = FCM.cmeans(data, int(np.rint(inst_num[num1])), 1.2, 0.1, 100, torch.cat(
            #     (data[0].reshape((1, data.shape[1])), data[-1].reshape(1, data.shape[1]))))
            # C * N
            # print(v)
            # print((v[0, 0] + v[1, 0]) / 2)
            # print(u)
            cnt1 = 0
            cnt2 = 0
            cnt3 = 0
            threshold = 0.5
            # mask only on foreground pixels
            # consider others to be bg
            # print(data.median(0))
            for i in range(N):
                # if data[i, :].sum() > data.median(0)[0].sum() + threshold:
                # if (v.sum() / 2 <= data[i, :].sum()):
                # mask_1[loc_xy[i][0], loc_xy[i][1]] = int(u[0, i] * 255)
                # mask_2[loc_xy[i][0], loc_xy[i][1]] = int(u[1, i] * 255)
                # mask_3[loc_xy[i][0], loc_xy[i][1]] = int(u[0, i] * 255)
                # mask_4[loc_xy[i][0], loc_xy[i][1]] = int(u[1, i] * 255)
                # continue
                # if u[0, i] >= 0.2 and u[0, i] <= 0.8:
                #     mask_1[loc_xy[i][0], loc_xy[i][1]] = int(255)
                #     mask_2[loc_xy[i][0], loc_xy[i][1]] = int(255)
                #     mask_3[loc_xy[i][0], loc_xy[i][1]] = int(255)
                #     mask_4[loc_xy[i][0], loc_xy[i][1]] = int(255)
                if u[0, i] <= threshold:
                    # cnt1 += 1
                    # if u[0, i-1] >= 1 - threshold and u[0, i+1] >= 1 - threshold:
                    #     mask_1[loc_xy[i][0], loc_xy[i][1]] = 120
                    #     mask_2[loc_xy[i][0], loc_xy[i][1]] = 60
                    #     mask_4[loc_xy[i][0], loc_xy[i][1]] = 180
                    # else:
                    mask_1[loc_xy[i][0], loc_xy[i][1]] = 60
                    mask_2[loc_xy[i][0], loc_xy[i][1]] = 120
                    mask_3[loc_xy[i][0], loc_xy[i][1]] = 180
                # # elif data[i, :].sum() < data.median(0)[0].sum() - threshold:
                # # elif (v.sum() / 2 > data[i, :].sum()):
                elif u[0, i] >= 1 - threshold:
                    # cnt2 += 1
                    # if u[0, i-1] <= threshold and u[0, i+1] <= threshold:
                    #     mask_1[loc_xy[i][0], loc_xy[i][1]] = 60
                    #     mask_2[loc_xy[i][0], loc_xy[i][1]] = 120
                    #     mask_3[loc_xy[i][0], loc_xy[i][1]] = 180
                    # else:
                    mask_1[loc_xy[i][0], loc_xy[i][1]] = 120
                    mask_2[loc_xy[i][0], loc_xy[i][1]] = 60
                    mask_4[loc_xy[i][0], loc_xy[i][1]] = 180
                else:
                    # print(2)
                    # cnt3 += 1
                    mask_1[loc_xy[i][0], loc_xy[i][1]] = mask_2[loc_xy[i][0], loc_xy[i][1]] = mask_3[
                        loc_xy[i][0], loc_xy[i][1]] = mask_4[loc_xy[i][0], loc_xy[i][1]] = 180
            # print(cnt1, cnt2, cnt3)
        else:
            for i in range(height):
                for j in range(width):
                    if target[:, i, j].sum() > 1:
                        mask_1[i, j] = mask_2[i, j] = mask_3[i, j] = mask_4[i, j] = 180
                    elif target[num1, i, j] == 1:
                        mask_1[i, j] = 60
                        mask_2[i, j] = 120
                        mask_3[i, j] = 180
                    elif target[num2, i, j] == 1:
                        mask_1[i, j] = 120
                        mask_2[i, j] = 60
                        mask_4[i, j] = 180
                    else:
                        mask_1[i, j] = mask_2[i, j] = mask_3[i, j] = mask_4[i, j] = 0
        return mask_1, mask_2, mask_3, mask_4, flag

    def module2model_dict(self, model_path):
        module_dict = torch.load(model_path, map_location='cpu')
        model_dict = {}
        for k, v in module_dict.items():
            model_dict[k[7:]] = v
        return model_dict


if __name__ == '__main__':
    transformer = Transformer(11)
    from segmentation.dataset.Dataset import DigitDataset

    dataset = DigitDataset('C:/Users/SRC/Downloads/SegDataset/number_segment_2/img',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_2/label',
                           'C:/Users/SRC/Downloads/SegDataset/number_segment_2/label.txt',
                           offset=25000)
    _, target, _ = dataset[0]
    print(target.shape)
    img = transformer.multi_hot2rgb(target)
    print(img)
    print(img.shape)
    img = Image.fromarray(img)
    img.show()
    # img = Image.open('C:/Users/SRC/Downloads/SegDataset/number_segment_2/img/img_25000.jpg')
    # img = np.array(img)
    # print(img)
    # print(img.shape)
    # print(img.dtype)
    # img = Image.fromarray(img)
    # img.show()
