#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()#神经网络模块存在两种模式，train模式（net.train())和eval模式（net.eval())。一般的神经网络中，这两种模式是一样的，只有当模型中存在dropout和batchnorm的时候才有区别。
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)  #batch_size默认为128
    l = len(data_loader)    #结果为469
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:  #-1表示使用cpu
            data, target = data.cuda(), target.cuda()#.cuda()转换成用gpu计算的张量形式（python中记录的笔记中有）
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item() #F.cross_entropy是求交叉熵的函数  得到损失的值
        #reduction：计算模式，可以为 none(逐个元素计算)，sum(所有元素求和，返回标量)，mean(加权平均，返回标量)
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]     #得到每个批次的预测结果，输出大小为128*1????????(gw写的)  # 值最大的那个即对应着分类结果，然后把分类结果保存在 y_pred 里
        #keepdim（bool）– 保持输出的维度 。当keepdim=False时，输出比输入少一个维度（就是指定的dim求范数的维度）。而keepdim=True时，输出与输入维度相同。
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()#view_as把 target 变成维度和 y_pred 一样的意思 # 将 y_pred 与 target 相比，得到正确预测结果的数量，并加到 correct 中
        #long()函数将数字或字符串转换为一个长整型。 cpu()不知道有啥用


    test_loss /= len(data_loader.dataset)   #除以数据集大小
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy#也可以再返回一个test_loss

