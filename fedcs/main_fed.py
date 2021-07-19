#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

def FedCS(the_idxs_users,train_time_list,upload_time_list,T_round):
    select_idxs_users=[]   #本轮选中的用户
    time_sum=0             #本轮时间，即是算法中的θ
    while len(the_idxs_users)>0:
        time_list=[max(train_time_list[i]-time_sum,0)+upload_time_list[i] for i in range(len(the_idxs_users))]
        time_min=min(time_list)                  #最小元素
        time_min_index=time_list.index(time_min) #最小元素的下标
        time_sum_temp=time_sum+time_min
        if time_sum_temp<=T_round:
            time_sum=time_sum_temp
            select_idxs_users.append(the_idxs_users[time_min_index])
            del the_idxs_users[time_min_index]  #选中后删除
            del train_time_list[time_min_index]
            del upload_time_list[time_min_index]
        else:
            break
    return select_idxs_users,time_sum


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')   #使用gpu或cpu
    # load dataset and split users
    if args.dataset == 'mnist':     #图片格式为28*28*1
        #Compose函数把多个图像处理步骤放在一起
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #均值和方差
        dataset_train = datasets.MNIST('../dataset/', train=True, download=True, transform=trans_mnist) # 华为云上删掉一个点，pytorch1.4
        dataset_test = datasets.MNIST('../dataset/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            #dict_users = mnist_iid(dataset_train, args.num_users)   #把数据集分成100份，即每份600个
            dict_users=np.load(f'./save/dict_users_{args.num_users}_L-{args.L}.npy',allow_pickle=True).tolist()
        else:
            #dict_users = mnist_noniid(dataset_train, args.num_users, args.L) #
            dict_users=np.load(f'./save/dict_users_{args.num_users}_L-{args.L}.npy',allow_pickle=True).tolist()
    elif args.dataset == 'cifar':   #图片格式为32*32*3
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../dataset/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../dataset/', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape    #结果为1*28*28

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob,'\n')     #打印神经网络信息，由nn.module类提供

    net_glob.load_state_dict(torch.load('./save/weight.pth'))

    net_glob.train()    #启用 BatchNormalization和Dropout, 与eval函数相对

    # copy weights
    w_glob = net_glob.state_dict()  #暂存初始网络参数

    # training
    loss_train = [] #存放每进行一次FedAvg的损失
    round_accuracy=[]
    user_num = []
    # cv_loss, cv_acc = [], []    #这行和下面4行参数都没用到
    # val_loss_pre, counter = 0, 0
    # net_best = None
    # best_loss = None
    # val_acc_list, net_list = [], []

    lr=args.lr #学习率
    gama=1  #信道分配
    B=1     #信道增益
    S=50    #模型大小
    p=1     #传输功率
    N0=1    #噪声功率
    #h_sq= np.abs(np.random.exponential(1,args.num_users).tolist())  #信道增益的平方,服从指数分布
    #all_upload_time_list=[int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间
    computer_level = np.random.uniform(1, 9, args.num_users).tolist()  #计算能力
    communicate_time = [] #记录每轮通信时间
    sigma=1/6  #训练时间修正因子
    #rest_users=range(args.num_users)
    T_round=1000 #每轮限制时间 设置1000/1500/2000
    
    acc_test=0
    iter=0
    acc=60
    # time_start=time.time() #计时开始
    #for iter in range(args.epochs):     #默认值已设为10,一个迭代进行一次FedAvg
    while (acc_test <= acc) and (iter < 30):
        print('Round {:3d}'.format(iter+1))
        w_locals, loss_locals = [], []  #存放每个用户的本地模型参数和损失
        m = max(int(args.frac * args.num_users), 1) #结果为10
        idxs_users = np.random.choice(range(args.num_users), m, replace=False).tolist()  #在100个用户中随机选10个用户
        h_sq= np.abs(np.random.exponential(1,args.num_users).tolist())  #信道增益的平方,服从指数分布，均值为50
        all_upload_time_list=[int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间
        #rest_users=list(set(rest_users)-set(idxs_users))
        #print('随机产生的用户为：',idxs_users)
        sample_account=[len(dict_users[i]) for i in idxs_users]  #样本数
        train_time_list=[int(sigma*args.local_ep*len(dict_users[i])/computer_level[i]) for i in idxs_users]      #本轮训练时间列表
        upload_time_list=[all_upload_time_list[i] for i in idxs_users]  #本轮上传时间列表
        # h_sq=np.abs(np.random.exponential(1,len(idxs_users)).tolist())  #信道增益的平方,服从指数分布
        # upload_time_list=[int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间
        print('样本数：',sample_account)
        print('上传时间：',upload_time_list)
        print('训练时间：',train_time_list)
        select_idxs_users, one_communicate_time=FedCS(idxs_users,train_time_list,upload_time_list,T_round)       #使用FedCS函数挑选用户,同时获得本轮通信时间
        print('挑选出的用户为：',select_idxs_users,'数量为',len(select_idxs_users))
        user_num.append(len(select_idxs_users))
        print('本轮通信时间：',one_communicate_time)
        for idx in select_idxs_users:
            # if acc_test>=98.5:
            #     lr = 0.005
            # elif acc_test>=98:
            #     lr = 0.01 
            # elif acc_test>=97:
            #     lr = 0.05
            # else:
            #     lr = 0.1
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], lr=lr) #idxs为一个用户的数据集，大小为600
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
        # update global weights

        w_glob = FedAvg(w_locals)

        # copy weight to net_glob   
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        round_accuracy.append(acc_test)                       #获取本轮精度
        communicate_time.append(one_communicate_time)         #获取本轮时间

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Average loss：{:.3f}'.format(loss_avg))
        print('Test accuracy：{:.2f}%'.format(acc_test),'\n')
        loss_train.append(loss_avg)
        iter=iter+1
    # time_end=time.time() #计时结束
    # print('time cost',time_end-time_start,'s')

    # plot loss curve
    Time=time.strftime("%m.%d.%H.%M", time.localtime()) #记录时间，用来画图的命名

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.xlabel('Round')
    plt.ylabel('Train_loss')
    plt.savefig('./figure/{}_Train loss_{}_IID L-{}_Epochs-{}_Accuracy-{}_T round-{}.png'.format(Time,args.model,args.L,iter,acc,T_round))

    # plot accuracy curve
    plt.figure()
    plot_x=[sum(communicate_time[:i+1]) for i in range(len(communicate_time))]  #计算横坐标
    plt.plot(plot_x,round_accuracy)
    plt.xlabel('Time / s')
    plt.ylabel('Test_accurac / %')
    plt.savefig('./figure/{}_Test accuracy_{}_IID L-{}_Epochs-{}_Accuracy-{}_T round-{}.png'.format(Time,args.model,args.L,iter,acc,T_round))

    # save data
    plot_data=[] #第一个保存训练损失，第二个保存时间，第三个保存测试精度，第四个保存用户数
    plot_data.append(loss_train)
    plot_data.append(plot_x)
    plot_data.append(round_accuracy)
    plot_data.append(user_num)
    np.save('./figure data/{}_Figure data_{}_IID L-{}_Epochs-{}_Accuracy-{}_T round-{}.npy'.format(Time,args.model,args.L,iter,acc,T_round),plot_data)
    print('数据保存成功')

    # # 华为云移动文件
    # import moxing as mox
    # mox.file.copy_parallel("./figure data","obs://federated--learning/fedcs/figure data")
