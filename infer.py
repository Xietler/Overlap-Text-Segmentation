# -*- coding: utf-8 -*-
# @Time    : 2019/3/14 19:47
# @Author  : Ruichen Shao
# @File    : infer.py

import sys
sys.path.insert(0, '/workspace/Light-head-rcnn')
import torch
from PIL import Image
from LightheadRCNN_Learner import LightHeadRCNN_Learner

def infer(model_path, image_path):
    learner = LightHeadRCNN_Learner(training=False)
    learner.load_state_dict(torch.load(str(model_path)))
    img = Image.open(image_path)
    bboxes, labels, scores = learner.predict_on_img(img, preset='detect', return_img=False, with_scores=False, original_size=True)
    return bboxes, labels, scores

if __name__ == '__main__':
    dict_mapping = {'0': 'layout', '1': 'cell', '2': 'table'}
    bboxes, labels, scores = infer('/workspace/Light-head-rcnn/work_space/final/model_2019-01-24-19-29_val_loss:0.8291515338420868_map05:0.9824390308094401_step:14000.pth', '/workspace/Light-head-rcnn/data/000001.jpg')
    for i in range(bboxes.shape[0]):
        print('bbox {} coordinates: {}'.format(i, bboxes[i]))
        print('label: {}'.format(dict_mapping[str(labels[i])]))
        print('score: {}'.format(scores[i]))