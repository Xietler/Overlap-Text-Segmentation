# -*- coding: utf-8 -*-
# @Time    : 2019/1/25 16:54
# @Author  : Ruichen Shao
# @File    : config.py

import os
class Config(object):
    def __init__(self):
        self.workspace = os.path.dirname(__file__)
