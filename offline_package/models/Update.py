#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#batch_size也就是args.local_bs，每一批加载多少个样本

    def train(self, net):#针对一个用户一轮的本地模型训练
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = [] #每次迭代的损失，一共迭代 args.local_ep=5 次，每次迭代遍历600个样本
        for iter in range(self.args.local_ep):#每个用户每轮本地训练模型时迭代5次
            batch_loss = []     #每个批次的损失，每个批次有50（local_bs）个样本，600/50=12 个批次
            for batch_idx, (images, labels) in enumerate(self.ldr_train):   #每个batch_idx大小为50，这里的循环为12次（就是12批次）
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)#交叉熵损失函数
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 11 == 0:   
                    #跑完1个批次就打印一次 就是跑完50个样本就打印一次
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

