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
import matplotlib.pyplot as plt
import numpy as np

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

def pack(c, w, v): #c是总通信时间上限（背包容量） w是每个用户的训练时间+上传时间 v是每个用户的二范数
    n = len(w)
    value = [[0 for j in range(c + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, c + 1):
            value[i][j] = value[i - 1][j]
            if j >= w[i - 1] and value[i][j] < value[i - 1][j - w[i - 1]] + v[i - 1]:
                value[i][j] = value[i - 1][j - w[i - 1]] + v[i - 1]
    return value,value[n][c]

def show(c, w, value):
    n = len(w)
    #print('背包总价值为:', value[n][c])
    x = [False for i in range(n)]
    j = c
    for i in range(n, 0, -1):
        if value[i][j] > value[i - 1][j]:
            x[i - 1] = True
            j -= w[i - 1]
    selected_user_order = []
    for i in range(n):
        if x[i]:
            selected_user_order.append(i+1)
    return selected_user_order

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')   #使用gpu或cpu

    # load dataset and split users
    if args.dataset == 'mnist':     #图片格式为28*28*1
        #Compose函数把多个图像处理步骤放在一起
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) #均值和方差
        dataset_train = datasets.MNIST('../dataset/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../dataset/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            #dict_users = mnist_iid(dataset_train, args.num_users)   #args.num_users=100=>把数据集分成100份（因为一共有100个用户），即每份600个  已经改成各个用户的数据成正太分布
            dict_users=np.load(f'./save/dict_users_{args.num_users}.npy',allow_pickle=True).tolist()
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':   #图片格式为32*32*3
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('C:/Users/86132/dataset/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('C:/Users/86132/dataset/cifar', train=False, download=True, transform=trans_cifar)
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
    print(net_glob)     #打印神经网络信息，由nn.module类提供

    net_glob.load_state_dict(torch.load('./save/weight.pth'))

    net_glob.train()    #启用 BatchNormalization和Dropout, 与eval函数相对

    # copy weights
    w_glob = net_glob.state_dict()  #暂存初始网络参数

    # training
    loss_train = [] #存放每进行一次FedAvg的损失
    cv_loss, cv_acc = [], []    #这行和下面4行参数都没用到
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []


    plot_x = []#绘制的图像的横轴
    acc_train_set = []#训练集上的准确率的集合
    round_accuracy = []#测试集上的准确率的集合

    gama = 1  #信道分配
    B = 1     #信道增益
    S = 50   #模型大小
    p = 1     #传输功率
    N0 = 1    #噪声功率
    sigma=1/3
    # h_sq = np.random.exponential(1,args.num_users)  #信道增益的平方,服从指数分布
    # upload_time = [int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间=模型大小/香农定理算出来的最大传输速率
    computer_level=np.random.uniform(1,9,args.num_users)  #计算能力 符合均匀分布，均值为5
    train_time = [int(sigma*args.local_ep * len(dict_users[i]) / computer_level[i]) for i in range(args.num_users)]  # 本轮训练时间列表 = 每个用户循环的次数（args.local_ep）乘用户的数据集大小/计算能力
    communicate_time=[] #记录每轮通信时间

    T_round = 1000 #每轮限制总时间 设置为1000/1500/2000

    time_start_sum = time.time()#记录所有轮数的起始训练时间
    m = max(int(args.frac * args.num_users), 1)  # 挑选10个用户
    #rest_user = list(range(args.num_users))  # 还没被挑选的用户 是10的整数倍
    #print(args.num_users)
    #每轮选10个用户
    acc_test=0
    iter=0
    acc=99
    # for iter in range(args.epochs):#共进行10轮
    while acc_test<=acc:
        print('Round {:3d}'.format(iter+1))
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)#本轮选择的用户的真实编号
        #print('idxs_users',idxs_users)
        delta_time = {}  # 用字典记录每个经过本地训练的用户的上传时间
        user_norm = {}   # 用字典记录一轮中所有用户的二范数
        w_locals, loss_locals = [], []      # 存放每个用户的本地模型参数和损失
        value_list = []  # 记录一轮中各个二范数的值
        one_epoch_actual_selected_user = [] # 一轮中达到拐点前每一个用户本地训练完后被选择的用户集的真实编号
        one_epoch_selected_user = []        # 一轮中达到拐点前每一个用户本地训练完后被选择的用户集的顺序编号
        cnt = 0          # 计数器
        h_sq = np.random.exponential(1,args.num_users)  #信道增益的平方,服从指数分布
        upload_time = [int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间=模型大小/香农定理算出来的最大传输速率

        selected_user_and_train_time = {}   # 一轮里被选中的用户及其训练时间
        for i in idxs_users:
            selected_user_and_train_time[i] = train_time[i]
        sorted_user = sorted(selected_user_and_train_time.items(), key=lambda kv: (kv[1], kv[0]))#按照训练时间（字典的value）排序 生成key为这一轮选择的10个用户的真实编号 value为这些用户对应的训练时间的字典
        train_order = [i[0] for i in sorted_user]#每轮用户的训练顺序 是他们的真实编号的集合
        local_train_time = []#参与本地训练的用户的训练时间的集合

        for idx in train_order:#一共循环10次 表示一轮中10个用户参与本地训练
            #LocalUpdate为一个用户一轮的训练网络的类  在Update.py文件中
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) #idxs为一个用户的数据集，大小为600
            ww, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            delta_time[idx] = upload_time[idx]  #上传时间
            local_train_time.append(train_time[idx])#参与本地训练的用户的训练时间的集合

            #求该用户二范数的变化量
            w=copy.deepcopy(ww)              #某用户训练完后的权值
            #print(w)
            delta_w=copy.deepcopy(w_glob)    #初始化Δw向量
            for Weight in delta_w.keys():
                delta_w[Weight] = w[Weight]-w_glob[Weight] #该用户训练完后的权值和全局模型权值做差
            w_norm = 0
            for w_name, w_par in delta_w.items():#一个用户会有多少个向量需要求二范数？8个，见myfed中的txt文件
                w_norm = w_norm+(np.linalg.norm(w_par.cpu().numpy()))**2 #(每个用户有很多个向量，每个向量都有一个二范数)依次求每个向量的二范数平方再求和  np.linalg.norm用来求二范数的函数
                #如果想把CUDA  float-tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式。 numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
            #print(len(dict_users[idx]),',用户',idx,':',(w_norm)**0.5)#w_norm是所有向量二范数的平方的和，最终每个用户的二范数是w_norm再开方#len(dict_users[idx])是idx号用户的数据集大小
            user_norm[idx] = (w_norm)**0.5
            #print(user_norm)

            w_locals.append(w)

            loss_locals.append(copy.deepcopy(loss))

            cnt += 1

            ######################################################
            # 每一个用户本地训练完就调用背包
            user_order = [key for key in delta_time]
            delta_time_list = [delta_time[key] for key in delta_time]
            user_norm_list = [user_norm[key] for key in user_norm]
            # print('当前进行完本地训练的总用户数', len(delta_time_list))
            pack_capacity = T_round-train_time[user_order[-1]]#当前背包总容量
            if pack_capacity <= 0:  
                print('本次没有进行背包。。')
                break

            returned_value_list, value = pack(pack_capacity, delta_time_list, user_norm_list)
            value_list.append(value)
            if value < max(value_list):
                print('背包总价值出现拐点，本轮结束')
                break
            selected_user_order = show(pack_capacity, delta_time_list, returned_value_list)  # 用于打印最大背包价值并返回选择的用户的顺序编号
            actual_selected_user = [user_order[i - 1] for i in selected_user_order]
            #print('本轮中，第',cnt,'个用户进行完本地训练后，背包算法选择的用户为:', actual_selected_user,'\n')#actual_selected_user是选择的用户的真实编号
            one_epoch_selected_user.append(selected_user_order)
            one_epoch_actual_selected_user.append(actual_selected_user)
            ############################################################################

        final_selected_user_order = one_epoch_selected_user[-1]  # 确定本轮拐点处选择的用户的顺序编号
        final_actual_selected_user = one_epoch_actual_selected_user[-1]# 确定本轮拐点处选择的用户的真实编号
        print('最终选中的用户为',final_actual_selected_user,'数量为',len(final_actual_selected_user))
        selected_upload_time = [upload_time[i] for i in final_actual_selected_user]#被选择的用户的上传时间的集合
        actual_w_locals = [w_locals[i - 1] for i in final_selected_user_order]  # 选择用户用于全局模型的更新 actual_w_locals记录被选择的用户的模型参数
        actual_loss_locals = [loss_locals[i - 1] for i in final_selected_user_order]  # 记录被选择的用户的损失
        time_sum = local_train_time[-1]+sum(selected_upload_time)#一轮的总通信时间是经过本地训练的用户的训练时间的最大值和被选中的用户的上传时间的总和
        print('本轮总用时',time_sum, 's')

        plot_x_gap = time_sum#要绘制的图像的相邻两点的时间差
        if iter == 0:
            plot_x.append(plot_x_gap)
        else:
            plot_x.append(plot_x[-1]+plot_x_gap)

        # update global weights 这是中心服务器进行的
        w_glob = FedAvg(actual_w_locals)#在Fed.py文件中

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # 1轮结束了， print loss
        loss_avg = sum(actual_loss_locals) / len(actual_loss_locals)
        # print('损失率：', loss_avg)
        loss_train.append(loss_avg)

        #确定准确率图像的纵轴的集合
        net_glob.eval()
        acc_test = test_img(net_glob, dataset_test, args)
        round_accuracy.append(acc_test)
        print('Average loss：{:.3f}'.format(loss_avg))
        print('Test accuracy：{:.2f}%'.format(acc_test.item()),'\n')
        net_glob.train()
        iter=iter+1

    # Time = time.strftime("%m.%d.%H.%M", time.localtime()) #记录时间，用来画图的命名

    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.xlabel('Round')
    # plt.ylabel('Train_loss')
    # plt.savefig('./figure/{}_Train loss_{}_Epochs-{}_T round-{}_accuracy-{}.png'.format(Time,args.model,iter,T_round,acc))#format中的参数对应于前面单引号中的大括号里面的内容

    # # plot accuracy curve
    # plt.figure()
    # # plot_x=[sum(communicate_time[:i+1]) for i in range(len(communicate_time))]  #计算横坐标
    # plt.plot(plot_x, round_accuracy)
    # plt.xlabel('Time / s')
    # plt.ylabel('Test_accuracy / %')
    # plt.savefig('./figure/{}_Test accuracy_{}_Epochs-{}_T round-{}_accuracy-{}.png'.format(Time,args.model,iter,T_round,acc))

    # # save data
    # plot_data=[] #第一个保存训练损失，第二个保存时间，第三个保存测试精度
    # plot_data.append(loss_train)
    # plot_data.append(plot_x)
    # plot_data.append(round_accuracy)
    # np.save('./figure_data/{}_Figure data_{}_Epochs-{}_T round-{}_accuracy-{}.npy'.format(Time,args.model,iter,T_round,acc),plot_data)
    # print('数据保存成功')

    # print("Testing accuracy: {:.2f}%".format(acc_test))

    Time=time.strftime("%m.%d.%H.%M", time.localtime()) #记录时间，用来画图的命名

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.xlabel('Round')
    plt.ylabel('Train_loss')
    plt.savefig('./figure/{}_Train loss_{}_IID-{}_Epochs-{}_T round-{}.png'.format(Time,args.model,args.iid,iter,T_round))

    # plot accuracy curve
    plt.figure()
    # plot_x=[sum(communicate_time[:i+1]) for i in range(len(communicate_time))]  #计算横坐标
    plt.plot(plot_x,round_accuracy)
    plt.xlabel('Time / s')
    plt.ylabel('Test_accurac / %')
    plt.savefig('./figure/{}_Test accuracy_{}_IID-{}_Epochs-{}_T round-{}.png'.format(Time,args.model,args.iid,iter,T_round))

    # save data
    plot_data=[] #第一个保存训练损失，第二个保存时间，第三个保存测试精度
    plot_data.append(loss_train)
    plot_data.append(plot_x)
    plot_data.append(round_accuracy)
    np.save('./figure data/{}_Figure data_{}_IID-{}_Epochs-{}_T round-{}.npy'.format(Time,args.model,args.iid,iter,T_round),plot_data)
    print('数据保存成功')





