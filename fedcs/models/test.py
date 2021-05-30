#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)  #batch_size默认为128
    l = len(data_loader)    #结果为469
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:  #-1表示使用cpu
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item() 
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]     #得到每个批次的预测结果，输出大小为128*1
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()    
        #long()函数将数字或字符串转换为一个长整型。 cpu()不知道有啥用

    test_loss /= len(data_loader.dataset)   #除以数据集大小
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

