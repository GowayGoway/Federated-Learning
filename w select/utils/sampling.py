#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


#对于独立同分布的数据，每个用户随机挑600张图片
# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users) #num_items=600
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


#对于独立同分布的数据，每个用户随机挑600张图片
def mnist_iid(dataset, num_users):
    num_items=[]    #存放每个用户的数据集下标
    num_items=np.abs(np.trunc(150*np.random.randn(num_users-1)+600).astype(int).tolist()) #随机生成用户的数据集个数，均值600，方差150
    dict_users, all_idxs = [], [i for i in range(len(dataset))]
    for i in range(num_users-1):
        dict_users.append(list(set(np.random.choice(all_idxs, num_items[i], replace=False)))) #在未挑中的数据集中随机挑
        all_idxs = list(set(all_idxs) - set(dict_users[i]))  #删去已经选中的数据
    dict_users.append((all_idxs))   #最后一个用户把剩下的数据挑走
    # np.save('C:/Users/86132/Desktop/dachuang/code/model1/save/dict_users.npy',dict_users)
    return dict_users


#对于非独立同分布的数据，将数据按标签排序，每个用户随机选两次300份图片，所以一个用户可以获得600张图片。的最终目的是将6万张图片分给100个用户，
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)} #创建一个大小为100的字典
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))   #vstack()按垂直方向（行顺序）堆叠数组构成一个新的数组,输出结果为2*60000
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] #将idxs_labels按标签（即第二行）排序，第一行为对应的下标
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))   #从idx_shard挑出两个数
        idx_shard = list(set(idx_shard) - rand_set)         #idx_shard减去刚刚挑的数
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)  #数组拼接
    return dict_users


# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def cifar_iid(dataset, num_users):
    num_items=[]    #每个用户的数据集个数
    num_items_average = int(len(dataset)/num_users)
    num_items=np.abs(np.trunc(num_items_average/4*np.random.randn(num_users-1)+num_items_average).astype(int).tolist()) #随机生成用户的数据集个数，均值600，方差150
    dict_users, all_idxs = [], [i for i in range(len(dataset))]
    for i in range(num_users-1):
        dict_users.append(list(set(np.random.choice(all_idxs, num_items[i], replace=False)))) #在未挑中的数据集中随机挑
        all_idxs = list(set(all_idxs) - set(dict_users[i]))  #删去已经选中的数据
    dict_users.append((all_idxs))   #最后一个用户把剩下的数据挑走
    #np.save('./save/cifar_dict_users.npy',dict_users)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('C:/Users/86132/dataset/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
