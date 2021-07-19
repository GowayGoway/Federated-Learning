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

def online_bag(value,weight,z):
    L = 0.1           #上界
    U = 70           #下界
    C = 1/(1+np.log(U/L))   #在[0,C]内都选
    if z <= C:      # #在线背包的选取阈值
        phi = L
    elif z<1:
        phi = ((U*np.e/L)**z)*(L/np.e)
    else:
        phi = U
    radio = value/weight
    if radio >= phi:
        return True
    else:
        return False

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
            #dict_users = mnist_iid(dataset_train, args.num_users)   #把数据集分成100份，即每份600个
            dict_users=np.load(f'./save/dict_users_{args.num_users}_L-{args.L}.npy',allow_pickle=True).tolist()
        else:
            # dict_users = mnist_noniid(dataset_train, args.num_users)
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
    print(net_glob)     #打印神经网络信息，由nn.module类提供

    net_glob.load_state_dict(torch.load('./save/weight.pth'))
    net_glob.train()    #启用 BatchNormalization和Dropout, 与eval函数相对
    # copy weights
    w_glob = net_glob.state_dict()  #暂存初始网络参数

    # training
    loss_train = [] #存放每进行一次FedAvg的损失
    round_accuracy=[]
    user_num =[] # 存放每进行一次FedAvg的用户数量

    lr=args.lr #学习率
    gama=1  #信道分配
    B=1     #信道增益
    S=100   #模型大小
    p=1     #传输功率
    N0=1    #噪声功率
    sigma=1/6 
    computer_level=np.random.uniform(1,9,args.num_users).tolist()  #计算能力
    communicate_time=[] #记录每轮通信时间
    T_round = 1000 #每轮限制时间 设置为1000/1500/2000

    #绘图坐标
    communicate_time = []#绘制的图像的横轴
    acc_train_set = []#训练集上的准确率的集合
    acc_test_set = []#测试集上的准确率的集合
    acc_test=0
    iter=0
    acc=95

    # time_start=time.time() #计时开始
    while (acc_test <= acc) and (iter < 30):     #训练达到指定精度停止
        h_sq= np.abs(np.random.exponential(1,args.num_users).tolist())  #信道增益的平方,服从指数分布
        all_upload_time_list=[int(S/(gama*B*np.log2(1+p*i/gama*B*N0))) for i in h_sq]   #上传时间
        print('Round {:3d}'.format(iter+1))
        w_locals, loss_locals = [], []  #存放每个用户的本地模型参数和损失
        w_norm_list = []            #存放每epoch训练用户的二范数
        time_traincost = []         #存放每个用户训练时间
        m = max(int(args.frac * args.num_users), 1) 
        idxs_users = [i for i in range(0,20)]
        sample_account=[len(dict_users[i]) for i in idxs_users]  #样本数
        train_time_list=[int(sigma*args.local_ep*len(dict_users[i])/computer_level[i]) for i in idxs_users]      #本轮训练时间列表
        upload_time_list=[all_upload_time_list[i] for i in idxs_users]  #本轮上传时间列表
        print('样本数：',sample_account)
        print('上传时间：',upload_time_list)
        print('训练时间：',train_time_list)
        
        #求二范数的过程
        # for idx in idxs_users:
        #     #LocalUpdate函数为一个用户的训练网络函数
        #     local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) #idxs为一个用户的数据集，大小为600
        #     ww, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

        #     #求该用户二范数的变化量
        #     w=copy.deepcopy(ww)              #某用户训练完后的权值
        #     delta_w=copy.deepcopy(w_glob)    #初始化Δw向量
        #     for Weight in delta_w.keys():
        #         delta_w[Weight]=w[Weight]-w_glob[Weight] #该用户训练完后的权值和全局模型权值做差
        #     w_norm=0
        #     for w_name,w_par in delta_w.items(): 
        #         w_norm=w_norm+(np.linalg.norm(w_par.cpu().numpy()))**2 #依次求二范数平方再求和
        #     w_norm_list.append(w_norm)
        #     # print(len(dict_users[idx]),',用户',idx,':',(w_norm)**0.5)     #二范数值

        #     w_locals.append(w)
        #     loss_locals.append(copy.deepcopy(loss))
        
        # #计算得到各用户的容量（△Ttrain+Tupdate）
        weight_time = [0 for _ in range(20)]       #存放用户的背包容量(深拷贝)
        t_temp = np.array(train_time_list)          #按训练时间排序
        index_t = np.argsort(t_temp)

        # #weight_time:物品重量   w_norm_list:对应的物品价值
        # w_norm_list = [i*1000 for i in w_norm_list]
        # w_norm_list = list(map(int,w_norm_list))

        #使用在线背包挑用户
        weight_time[index_t[0]] = train_time_list[index_t[0]] + upload_time_list[index_t[0]]
        for i in range(1,m):
            weight_time[index_t[i]] = train_time_list[index_t[i]]-train_time_list[index_t[0]] + upload_time_list[index_t[i]]    
        select_client = []
        bag_capacity = T_round             #背包总容量
        z = 0                           #已装背包重量
        queue_time = 0
        cur_train_time = train_time_list[index_t[0]]
        for ind in index_t:
            zj = z/bag_capacity         #归一化已装重量
            #更新重量
            if train_time_list[ind]>=z:  #当前用户训练时间大于等于已用的时间(背包容量)
                weight_time[ind] = train_time_list[ind] - z + upload_time_list[ind] 

            else:                       ##当前用户训练时间小于已用的时间(背包容量)
                weight_time[ind] = upload_time_list[ind]    #重量只有上传时间

            #求二范数过程
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[ind], lr=lr) #idxs为一个用户的数据集，大小为600
            ww, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #求该用户二范数的变化量
            w=copy.deepcopy(ww)              #某用户训练完后的权值
            delta_w=copy.deepcopy(w_glob)    #初始化Δw向量
            for Weight in delta_w.keys():
                delta_w[Weight]=w[Weight]-w_glob[Weight] #该用户训练完后的权值和全局模型权值做差
            w_norm=0
            for w_name,w_par in delta_w.items(): 
                w_norm=w_norm+(np.linalg.norm(w_par.cpu().numpy()))**2 #依次求二范数平方再求和
            w_norm = int(w_norm*1000)
            if train_time_list[ind] > bag_capacity:
                break
            #在线背包选取过程
            if online_bag(w_norm,weight_time[ind],zj):       #判断是否选
                if train_time_list[ind] > z :       #目前选中用户训练时间大于已用背包时间--不用排队
                    temp = z
                    z = train_time_list[ind]+upload_time_list[ind] #更新已用背包容量
                    if z<=bag_capacity:     #背包塞得下
                        select_client.append(ind)
                        cur_train_time = train_time_list[ind]       #更新最新的最大用户训练时间
                        w_norm_list.append(w_norm)
                        w_locals.append(w)
                        loss_locals.append(copy.deepcopy(loss))
                    else:               #背包塞不下
                        z = train_time_list[ind]            #已用背包容量回退
                else:                   #目前选中用户训练时间小于已用背包时间--排队
                    temp = z
                    z = temp+upload_time_list[ind] #更新已用背包容量
                    if z<=bag_capacity:     #背包塞得下
                        select_client.append(ind)
                        cur_train_time = train_time_list[ind]       #更新最新的最大用户训练时间
                        w_norm_list.append(w_norm)
                        w_locals.append(w)
                        loss_locals.append(copy.deepcopy(loss))
                    else:               #背包塞不下
                        z = temp            #已用背包容量回退
            else:
                if train_time_list[ind]>z:  #用户训练时间大于已装背包容量
                    if train_time_list[ind]<=bag_capacity:
                        z = train_time_list[ind]
                        continue
                    else:           #该用户训练时间超过背包容量
                        break
                else:
                    continue
        
        print('挑选的用户为：',select_client,'数量为：',len(select_client))
        user_num.append(len(select_client))
        print('本轮所耗费时间:%d'%(z))

        try:
            # update global weights
            w_glob = FedAvg(w_locals)
        except:
            print('本轮未选中用户')

        # copy weight to net_glob   
        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        net_glob.train()
        round_accuracy.append(acc_test)                       #获取本轮精度

        # print loss
        try:
            loss_avg = sum(loss_locals) / len(loss_locals)
        except:
            loss_avg = loss_avg
        print('Average loss：{:.3f}'.format(loss_avg))
        print('Test accuracy：{:.2f}%'.format(acc_test),'\n')
        loss_train.append(loss_avg)
 
        communicate_time.append(z)
        iter = iter + 1

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
    plot_data=[] #第一个保存训练损失，第二个保存时间，第三个保存测试精度
    plot_data.append(loss_train)
    plot_data.append(plot_x)
    plot_data.append(round_accuracy)
    plot_data.append(user_num)
    np.save('./figure data/{}_Figure data_{}_IID L-{}_Epochs-{}_Accuracy-{}_T round-{}.npy'.format(Time,args.model,args.L,iter,acc,T_round),plot_data)
    print('数据保存成功')

    # # 华为云移动文件
    # import moxing as mox
    # mox.file.copy_parallel("./figure data","obs://federated--learning/online/figure data")



