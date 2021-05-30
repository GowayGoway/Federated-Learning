#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def FedAvg(w):#一轮训练结束才会调用一次这个函数
    w_avg = copy.deepcopy(w[0])#w[0]是所有的矩阵 以OrderedDict形式存在
    for k in w_avg.keys():      #以列表形式（并非直接的列表，若要返回列表值还需调用list函数）返回字典中的所有的键。
        for i in range(1, len(w)):#len(w)是8，表示有8个向量
            w_avg[k] += w[i][k]#w[i][k]是key对应的所有tensor矩阵
        w_avg[k] = torch.div(w_avg[k], len(w))  #w_avg[k]逐个除以len(w)
    return w_avg
