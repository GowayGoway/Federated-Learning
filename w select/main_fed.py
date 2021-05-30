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


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')   #使用gpu或cpu

    # load dataset and split users
    if args.dataset == 'mnist':     #图片格式为28*28*1
        #Compose函数把多个图像处理步骤放在一起
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #均值和方差
        dataset_train = datasets.FashionMNIST('../dataset/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('../dataset/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            #dict_users = mnist_iid(dataset_train, args.num_users)   #把数据集分成100份，即每份600个
            dict_users=np.load(f'./save/dict_users_{args.num_users}.npy',allow_pickle=True).tolist()
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':   #图片格式为32*32*3
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../dataset/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../dataset/', train=False, download=True, transform=trans_cifar)
        if args.iid:
            #dict_users = cifar_iid(dataset_train, args.num_users)
            dict_users=np.load(f'./save/cifar_dict_users_{args.num_users}.npy',allow_pickle=True).tolist()
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

    #torch.save(net_glob.state_dict(),'./save/cifar_weight.pth')

    # copy weights
    # net_glob.load_state_dict(torch.load('./save/cifar_weight.pth'))
    net_glob.load_state_dict(torch.load('./save/weight.pth'))
    net_glob.train()    #启用 BatchNormalization和Dropout, 与eval函数相对
    w_glob = net_glob.state_dict()  #暂存初始网络参数

    # training
    loss_train = [] #存放每进行一次FedAvg的损失
    round_accuracy=[]

    gama=1  #信道分配
    B=1     #信道增益
    S=100   #模型大小
    p=1     #传输功率
    N0=1    #噪声功率
    sigma=1/3  #训练时间修正因子
    # h_sq= np.abs(np.random.exponential(1,args.num_users).tolist())  #信道增益的平方,服从指数分布
    # all_upload_time_list=[int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间
    computer_level=np.random.uniform(1,9,args.num_users).tolist()  #计算能力
    communicate_time=[] #记录每轮通信时间    
    all_train_time_list=[int(sigma*args.local_ep*len(dict_users[i])/computer_level[i]) for i in range(args.num_users)]
    #upload_time_list = np.load('./save/upload_time.npy').tolist()
    #upload_time_list= np.abs(np.trunc(15*np.random.randn(args.num_users)+60).astype(int).tolist()) #随机设置上传时间
    acc_test=0
    iter=0
    acc=70
    # np.save('./save/upload_time.npy',upload_time_list)
    # time_start=time.time() #计时开始
    #for iter in range(args.epochs):     #默认值已设为10,一个迭代进行一次FedAvg
    while acc_test<=acc:
        print('Round {:3d}'.format(iter+1))
        w_locals, loss_locals,w_idx= [], [], []#存放每个用户的本地模型参数和损失
        w_delta=[]      #存放每个用户的||Δw||
        m = max(int(args.frac * args.num_users), 1) #结果为10
        h_sq= np.abs(np.random.exponential(1,args.num_users).tolist())  #信道增益的平方,服从指数分布
        all_upload_time_list=[int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  #在100个用户中随机选10个用户
        train_time=max([all_train_time_list[idx] for idx in idxs_users])
        for idx in idxs_users:
            #LocalUpdate函数为一个用户的训练网络函数
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) #idxs为一个用户的数据集，大小为600
            ww, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            #求该用户二范数的变化量
            w=copy.deepcopy(ww)              #某用户训练完后的权值
            delta_w=copy.deepcopy(w_glob)    #初始化Δw向量
            for Weight in delta_w.keys():
                delta_w[Weight]=w[Weight]-w_glob[Weight] #该用户训练完后的权值和全局模型权值做差
            w_norm=0
            for w_name,w_par in delta_w.items(): 
                w_norm=w_norm+(np.linalg.norm(w_par.cpu().numpy()))**2 #依次求二范数平方再求和
            #print(len(dict_users[idx]),',用户',idx,':',(w_norm)**0.5)
            #
            w_locals.append(w)
            loss_locals.append(copy.deepcopy(loss))
            w_delta.append(w_norm)
        # np.save('./save/w_delta.npy',w_delta)
        # update global weights
        #挑选二范数大的一半用户
        sorted_list=sorted(enumerate(w_delta),key=lambda x: x[1],reverse = True)
        w_idx=[idxs_users[i[0]] for i in sorted_list][0:int(m/2)]
        upload_time=sum([all_upload_time_list[i] for i in w_idx])
        w_locals=[w_locals[i[0]] for i in sorted_list][0:int(m/2)]
        one_communicate_time=train_time+upload_time
        communicate_time.append(one_communicate_time)
        #print('选中的用户为',w_idx)
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob   
        net_glob.load_state_dict(w_glob)
        #if iter%5==0:s
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        round_accuracy.append(acc_test)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Average loss {:.3f}'.format(loss_avg))
        print('Test accuracy {:.2f}%'.format(acc_test))
        #print('训练时间：',train_time,',上传时间：',upload_time,'\n')
        loss_train.append(loss_avg)
        iter=iter+1
    # time_end=time.time() #计时结束
    # print('time cost',time_end-time_start,'s')

    Time=time.strftime("%m.%d.%H.%M", time.localtime()) #记录时间，用来画图的命名
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.xlabel('Round')
    plt.ylabel('Train_loss')
    plt.savefig('./figure/{}_Train loss_{}_IID-{}_Epochs-{}.png'.format(Time,args.model,args.iid,iter))

    # plot accuracy curve
    plt.figure()
    plot_x=[sum(communicate_time[:i+1]) for i in range(len(communicate_time))]  #计算横坐标
    plt.plot(plot_x,round_accuracy)
    plt.xlabel('Time / s')
    plt.ylabel('Test_accurac / %')
    plt.savefig('./figure/{}_Test accuracy_{}_IID-{}_Epochs-{}.png'.format(Time,args.model,args.iid,iter))

    # save data
    plot_data=[] #第一个保存训练损失，第二个保存时间，第三个保存测试精度
    plot_data.append(loss_train)
    plot_data.append(plot_x)
    plot_data.append(round_accuracy)
    np.save('./figure data/{}_Figure data_{}_IID-{}_Epochs-{}.npy'.format(Time,args.model,args.iid,iter),plot_data)
    print('数据保存成功')
