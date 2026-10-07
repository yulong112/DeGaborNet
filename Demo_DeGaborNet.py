from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import numpy as np
import torch
from torch import nn, optim
import os
import PIL
import PIL.Image
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.io import loadmat
import pandas as pd
import time
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random
from evaluation import AA_andEachClassAccuracy, metrics, show_results, predictv5
from zdataset import *
from DeGaborNet_all_models import *
from lightSpectrumDataset import splitTrainTestIndex, LightSpectrumDataset
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from thop import profile
# from torchstat import stat
import os
import random
import numpy as np
import torch
import gc
from dataProcess import zeroPadding, coordinates_to_mask, add_gaussian_noise

# 基础参数的设置
def settings(args):

    if args.data_name == 'Houston2018':
        args.n_cols = 2384  # number of col umns
        args.n_rows = 601  # number of rows
        args.n_channels = 48  # number of bands
        args.dim = 15  # patch size
        args.filter_size = 5  # filter size
        args.n_classes = 20  # number of predefined classes
        args.n_perclass = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50, 50]  # number of training samples per class
        # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
        # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
        # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
        # args.n_validate = 20 - 50  # number of validation samples per class
        args.n_validate = 20.18 - 200  # number of validation samples per class
        # args.mk = 1  # whether make data augmentation
        # args.id_set = 2  # the index of training sets
        # args.bnum = 3  # number of convolutional blocks

    elif args.data_name == 'PaviaU':
        args.n_channels = 103  # number of bands
        args.dim = 15 # patch size
        args.filter_size = 5  # filter size
        args.n_classes = 9  # number of predefined classes
        args.n_perclass = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50, 50]  # number of training samples per class
        # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
        # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
        # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
        args.n_validate = 9 - 50  # number of validation samples per class
        # # args.mk = 1  # whether make data augmentation
        # args.id_set = 2  # the index of training sets
        # args.bnum = 3  # number of convolutional blocks

    elif args.data_name == 'WHU_Hi_HongHu':
        args.n_cols = 475  # number of col umns
        args.n_rows = 940  # number of rows
        args.n_channels = 270  # number of bands
        args.dim = 15  # patch size
        args.filter_size = 5  # filter size
        args.n_classes = 22  # number of predefined classes
        args.n_perclass = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50,
                           50]  # number of training samples per class
        # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
        # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
        # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
        args.n_validate = 22 - 30  # number of validation samples per class
        # args.mk = 1  # whether make data augmentation
        # args.id_set = 2  # the index of training sets
        # args.bnum = 4  # number of convolutional blocks

    elif args.data_name == 'Chikusei':
        args.n_cols = 2335  # number of col umns
        args.n_rows = 2517  # number of rows
        args.n_channels = 128  # number of bands
        args.dim = 15  # patch size
        args.filter_size = 5  # filter size
        args.n_classes = 19  # number of predefined classes
        args.n_perclass = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50,
                           50]  # number of training samples per class
        # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
        # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
        # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
        args.n_validate = 19 - 50  # number of validation samples per class
        # args.mk = 1  # whether make data augmentation
        # args.id_set = 2  # the index of training sets
        # args.bnum = 4  # number of convolutional blocks

    elif args.data_name == 'WHU_Hi_HanChuan':
        args.n_cols = 303  # number of col umns
        args.n_rows = 1217  # number of rows
        args.n_channels = 274  # number of bands
        args.dim = 15  # patch size
        args.filter_size = 5  # filter size
        args.n_classes = 19  # number of predefined classes
        args.n_perclass = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50,
                           50]  # number of training samples per class
        # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
        # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
        # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
        args.n_validate = 16 - 30  # number of validation samples per class
        # args.mk = 1  # whether make data augmentation
        # args.id_set = 2  # the index of training sets
        # args.bnum = 4  # number of convolutional blocks

    elif args.data_name == 'MatiwanVillage':
        args.n_cols = 3750  # number of col umns
        args.n_rows = 1580  # number of rows
        args.n_channels = 256  # number of bands
        args.dim = 15  # patch size
        args.filter_size = 5  # filter size
        args.n_classes = 20  # number of predefined classes
        args.n_perclass = [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50,
                           50]  # number of training samples per class
        # [33,100,100,100,100,100,20,100,14,100,100,100,100,100,100,75] 1342
        # [33,50,50,50,50,50,20,50,14,50,50,50,50,50,50,50] 717
        # [33,200,200,181,200,200,20,200,14,200,200,200,143,200,200,75] 2466
        args.n_validate = 20 - 200  # number of validation samples per class
        # args.mk = 1  # whether make data augmentation
        # args.id_set = 2  # the index of training sets
        # args.bnum = 3  # number of convolutional blocks

    ###########################################################################

    # default options
    if args.default_settings:
        args.n_epochs = 300
        args.batch_size = 50
        args.learning_rate = 0.0076#0.0076
        args.std_mult = 0.4
        args.delay = 120
        args.n_theta = 4
        args.n_omega = 4
        args.gains = 2
        args.lr_div = 10
        args.even_initial = True

    return args
    # end of settings


def main(args):

    ## Manually adjustable parameters

    # args.data_name = 'Houston2018'
    # args.data_name = 'Chikusei'
    args.data_name = 'PaviaU'
    # args.data_name = 'MatiwanVillage' ## set the HSI dataset

    args.dataset_path = 'dataset/'
    model_name = 'DeGaborNet'   ## Use 'DeGaborNet-R' (with the advantage of noise robustness) or 'DeGaborNet' (regular mode)
    use_fast_inference_mode = 1   ## use_fast_inference_mode = 1 (fast inference version); use_fast_inference_mode = 0 (fast training version)
    add_noise_flag = 0            ## add gaussian noise to the HSI data
    set_noise_level = 0.4

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    F1score_ALL = []
    Train_Time_ALL = []
    Patchwise_Inference_Time_ALL = []
    Patchwise_Inference_forwardTime_ALL = []
    OA_ALL2 = []
    AA_ALL2 = []
    KPP_ALL2 = []
    AVG_ALL2 = []
    Fullimage_Inference_Time_ALL = []
    Fullimage_Inference_forwardTime_ALL = []
    for seed in range(5):
        # seed=1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ##### SETUP AND LOAD DATA #####
        args = settings(args)

        ###############################
        # 加载数据集
        X, y = loadData(args.data_name)
        if add_noise_flag==1:
            X = add_gaussian_noise(X, nSig=set_noise_level)

        X = normalization(X)
        channels = args.n_channels
        net_dim = 2
        # X,channels = applyPCA(X, numComponents=75)
        # print('Data shape after PCA: ', X,channels)

        exp_sample_amount_dict = {  # 用于构建train集的样本数量
            19: [1000, 1000, 140, 2000, 1500, 500, 2000, 2000, 2000, 600, 2000, 1000, 500, 2000, 200, 100, 500, 400, 100],
            19.1: [10, 10, 20, 50, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50, 30, 30, 30, 10, 10],
            19 - 100: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            19 - 50: [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            19 - 10: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            19 - 400: [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400],
            9 - 200: [200, 200, 200, 200, 200, 200, 200, 200, 200],
            9 - 100: [100, 100, 100, 100, 100, 100, 100, 100, 100],
            9 - 50: [50, 50, 50, 50, 50, 50, 50, 50, 50],
            9 - 30: [30, 30, 30, 30, 30, 30, 30, 30, 30],
            9 - 20: [20, 20, 20, 20, 20, 20, 20, 20, 20],
            9 - 10: [10, 10, 10, 10, 10, 10, 10, 10, 10],
            # 14 - 45: [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 45],
            14 - 7: [130, 50, 150, 100, 130, 130, 130, 100, 150, 120, 150, 90, 100, 45],
            14: [120, 45, 120, 100, 130, 130, 120, 100, 150, 120, 150, 80, 130, 45],
            15 - 100: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            15 - 50: [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            15 - 30: [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
            16 - 5: [5, 140, 80, 24, 30, 70, 5, 40, 10, 100, 225, 60, 20, 120, 45, 25],
            16 - 10: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            16 - 30: [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
            16 - 50: [33, 50, 50, 50, 50, 50, 20, 50, 14, 50, 50, 50, 50, 50, 50, 50],
            16 - 100: [33, 100, 100, 100, 100, 100, 20, 100, 14, 100, 100, 100, 100, 100, 100, 75],
            16 - 200: [33, 200, 200, 181, 200, 200, 20, 200, 14, 200, 200, 200, 143, 200, 200, 75],
            16 - 300: [33, 300, 300, 181, 300, 300, 20, 300, 14, 300, 300, 300, 143, 300, 300, 75],
            16 - 400: [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400],
            16 - 4100: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            16 - 450: [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            20 - 500: [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
            20 - 400: [400, 400, 400, 400, 400, 400, 200, 400, 400, 400, 400, 400, 400, 400, 400, 400, 100, 400, 400, 400],
            20 - 50: [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
            20 - 100: [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            20.18 - 200: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 200, 200,
                          200],
            20 - 200: [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
                       200],
            22 - 10: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            22 - 30: [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
            100: [400, 100, 100, 100, 80, 100, 50, 200, 200, 200, 200, 200, 200],  # ksc
            7.18: [700, 1000, 900, 13, 1000, 1000, 2000],  # HS18
            7.13: [150, 150, 150, 140, 150, 200, 200],  # HS13
            13: [300, 100, 100, 100, 50, 100, 50, 200, 200, 200, 200, 200, 400]
        }
        # 取到训练集\测试集坐标
        train_index, test_index = splitTrainTestIndex(y, trainNumList=exp_sample_amount_dict[args.n_validate],
                                                      removeZeroLabels=True)
        train_loader = LightSpectrumDataset(X, y, index=train_index, window_size=args.dim, use_lstm=False,
                                            dim=net_dim, removeZeroLabels=True)  # 实例化该类，将训练集的数据准备好
        # test_loader = LightSpectrumDataset(X, y, index=test_index, window_size=args.dim, use_lstm=False,
        #                                    dim=net_dim, removeZeroLabels=True)  # 实例化该类，将测试集的数据准备好
        test_dataset = LightSpectrumDataset(X, y, index=test_index, window_size=args.dim, use_lstm=False,
                                            dim=net_dim, removeZeroLabels=True)  # 实例化该类，将测试集的数据准备

        y_test = test_dataset.getY()  # 读取测试集的标签

        #
        train_loader = DataLoader(dataset=train_loader, batch_size=50, shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False, pin_memory=True,
                                 drop_last=False)

        #######################################
        # 训练
        # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        # 网络,损失函数,优化器
        # 网络放到GPU上
        if use_fast_inference_mode==1:
            if model_name=='DeGaborNet':
                net = DeGaborNet_fast_inference_mode(in_channels=channels, num_classes=args.n_classes, kernel=args.filter_size, model_name=0).to(device)
            elif model_name=='DeGaborNet-R':
                net = DeGaborNet_fast_inference_mode(in_channels=channels, num_classes=args.n_classes, kernel=args.filter_size, model_name=1).to(device)
            net = torch.jit.script(net)
            net_name = model_name+"_fast_inference_mode"
        else:
            net = DeGaborNet_fast_training_mode(in_channels=channels, num_classes=args.n_classes, kernel=args.filter_size, model_name=model_name).to(device)
            net_name = model_name+"_fast_training_mode"

        print('Model size: {:0.2f} float parameters'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        criterion = nn.CrossEntropyLoss()  # 损失函数：计算模型的输出与真实标签之间的交叉熵损失
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)  # 1DXY
        # optimizer = optim.Adam(net.parameters(), lr=0.001,weight_decay=3e-4)  # SSTN
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate)
        lr_lambda = lambda epoch: 0.1 ** ((epoch - 1) / 50) if epoch > 1 else 1.  # 动态学习率，随着epoch增加不断变化
        # lr_lambda = lambda epoch: 0.995 *
        # * epoch
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)

        # 模型输入需要的形状[batch,in_channels,h,w]
        # 开始训练
        train_loss = np.zeros(args.n_epochs)
        _loss = []
        accuracy = []
        acc = []
        net.train()  # 切换到训练模式，准备训练
        train_start_time = time.perf_counter()
        all_forward_time = 0
        all_backward_time = 0
        all_step_time = 0
        # scaler = GradScaler()
        # 定义三对 Event
        start_f = torch.cuda.Event(enable_timing=True)
        end_f = torch.cuda.Event(enable_timing=True)
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)
        start_s = torch.cuda.Event(enable_timing=True)
        end_s = torch.cuda.Event(enable_timing=True)
        for epoch in range(args.n_epochs):
            total_loss = 0  # 定义损失值
            tbar = tqdm(train_loader)
            total_correct = 0.  # 正确样本数
            count = 0
            e_forward_time = 0  # 用于累计正向传播的时间
            e_backward_time = 0  # 用于累计反向传播的时间
            e_step_time = 0  # 用于累计梯度更新的时间
            time_train_start = time.perf_counter()
            for i, (inputs, labels) in enumerate(tbar):  # train_loader训练集              #？？？？？
                # enumerate()用于可迭代\可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标.（例如将i和（inputs，labels）组成一个索引序列）
                inputs = inputs.to(device)  # 数据在GPU上运行
                labels = labels.to(device)
                # 优化器梯度归零
                optimizer.zero_grad()  # 将梯度归零
                # 正向传播 +　反向传播 + 优化1
                # with autocast():
                # torch.cuda.synchronize(device)
                # start_time1 = time.perf_counter()  # 记录开始时间
                start_f.record()
                outputs = net(inputs)  # 正向传播
                end_f.record()
                end_f.synchronize()
                e_forward_time += (start_f.elapsed_time(end_f) / 1000)
                # torch.cuda.synchronize(device)
                # end_time1 = time.perf_counter()  # 记录结束时间
                # e_forward_time += (end_time1 - start_time1)  # 累计正向传播时间
                loss = criterion(outputs, labels)  # 损失
                # torch.cuda.synchronize(device)
                # start_time2 = time.perf_counter()  # 记录开始时间
                start_b.record()
                loss.backward()  # 反向传播得到每个参数的梯度
                # torch.cuda.synchronize(device)
                # end_time2 = time.perf_counter()  # 记录结束时间
                # e_backward_time += (end_time2 - start_time2)
                end_b.record()
                end_b.synchronize()
                e_backward_time += (start_b.elapsed_time(end_b) / 1000)
                # torch.cuda.synchronize(device)
                # start_time3 = time.perf_counter()  # 记录开始时间
                start_s.record()
                optimizer.step()  # 通过梯度下降一步参数更新。进行单次优化（一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。）
                # torch.cuda.synchronize(device)
                # end_time3 = time.perf_counter()  # 记录结束时间
                # e_step_time += (end_time3 - start_time3)
                end_s.record()
                end_s.synchronize()
                e_step_time += (start_s.elapsed_time(end_s) / 1000)
                # 打印batch级别日记
                total_loss += loss.item()
                total_correct += torch.sum(torch.max(outputs, 1)[1] == labels.data)
                count += labels.shape[0]
                tbar.set_description(
                    f'Stage ({epoch + 1}/{args.n_epochs}) loss: {total_loss / (i + 1)} acc: {total_correct / count}')
            # concatenated_tensor = torch.cat(e_inputs, dim=0)
            # print(f"Shape after concatenation: {concatenated_tensor.shape}")
            all_forward_time += e_forward_time
            all_backward_time += e_backward_time
            all_step_time += e_step_time
            lr_schedule.step()
            # accuracy.append(total_correct / count)
            train_loss_ = total_loss / len(train_loader)
            train_loss[epoch] = train_loss_
            acc.append(total_correct / count)
            # print(train_loss_)
            # print(acc[-1])
            accuracy.append(acc[-1].tolist())
            # print(accuracy)
            _loss.append(train_loss_)

            '''
            print('[Epoch: %d]   [loss avg: %.4f]   [acc: %.4f]' % (
                epoch + 1, total_loss / len(train_loader), total_correct/count))  # 第几轮训练、平均损失、当前损失
            '''
            train_end_time = time.perf_counter()
            print('epoch train cost %.4fs' % (train_end_time - time_train_start))
        train_end_time = time.perf_counter()
        print('train cost %.4fs' % (train_end_time - train_start_time))
        Training_Time = all_forward_time+all_backward_time+all_step_time
        print(f'forward_time: {all_forward_time:.4f}s')
        print(f'backward_time: {all_backward_time:.4f}s')
        print(f'step_time: {all_step_time:.4f}s')
        print('Finished Training')
        # print(net.state_dict().keys())  # 打印模型参数
        # 保存模型参数
        # save_root = f"{net_name}"
        # if not os.path.exists(save_root):
        #     os.makedirs(save_root)
        # torch.save(net.state_dict(), "./model/net_parameter" + args.data_name + ".pkl")
        # torch.save(net, f"{net_name}/net_parameter" + args.data_name + str(args.filter_size) + str(args.bnum) + ".pkl")

        #####################################
        # 测试
        # 加载模型参数
        # net = DeGaborNet_model(in_channels=args.n_channels, num_classes=args.n_classes).to(device)
        # net.load_state_dict(torch.load("./model/net_parameter" + args.data_name + ".pkl"))
        net.eval()
        valid_mask = coordinates_to_mask(test_index, y.shape)
        window_size = args.dim

        ################### Patch-wise Inference ##################
        margin = window_size // 2
        height, width = y.shape[:2]

        X_input = zeroPadding(X, margin)
        Patchwise_Inference_startT = time.perf_counter()
        classification_map,Patchwise_Inference_forwardTime = predictv5(X_input, y, window_size, net, height, width, net_dim, device, 150)
        Patchwise_Inference_endT = time.perf_counter()
        Patchwise_Inference_allTime = Patchwise_Inference_endT - Patchwise_Inference_startT
        # 提取有效像素的预测和真实标签
        y_pred_test = classification_map[valid_mask]-1
        y_test = y[valid_mask]-1
        # 计算准确率
        correct = (y_pred_test == y_test).sum()
        total = valid_mask.sum()
        oa = correct / total if total > 0 else 0.0

        # 2. 清理 CPU 内存
        gc.collect()
        # 3. 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()


        ################### Full-image Inference ##################
        inputs2 = test_dataset.features
        inputs2 = torch.from_numpy(inputs2.astype(np.float32)).unsqueeze(0)
        inputs2 = inputs2.to(device)
        # 模型测试
        test_start_time2 = time.perf_counter()
        Fullimage_Inference_forwardTime = 0  # 用于累计test_net的时间
        start_t = torch.cuda.Event(enable_timing=True)
        end_t = torch.cuda.Event(enable_timing=True)
        start_t.record()
        with torch.no_grad():
            outputs = net(inputs2, is_training_patch=False, patch_size=window_size)
        end_t.record()
        end_t.synchronize()
        Fullimage_Inference_forwardTime += (start_t.elapsed_time(end_t) / 1000)
        # 4. 去除padding
        out_cropped = outputs.detach().cpu().numpy()    # [1, H, W, num_classes]
        # 5. 获取预测类别
        y_pred_ALL2 = np.argmax(out_cropped, axis=-1)  # [1, H, W]
        y_pred_ALL2 = y_pred_ALL2.squeeze(0)  # [H, W]
        test_end_time2 = time.perf_counter()
        Fullimage_Inference_allTime = test_end_time2 - test_start_time2

        # 6. 计算OA（只考虑有效像素）
        # 提取有效像素的预测和真实标签
        y_pred_valid = y_pred_ALL2[valid_mask]
        y_true_valid = y[valid_mask]-1
        # 计算准确率
        correct = (y_pred_valid == y_true_valid).sum()
        total = valid_mask.sum()
        oa2 = correct / total if total > 0 else 0.0

        # 分类报告
        classification = classification_report(y_test, y_pred_test, digits=4)
        # 混淆矩阵
        confusion = confusion_matrix(y_test, y_pred_test)
        # 每类准确率和平均准确率
        each_acc, aa = AA_andEachClassAccuracy(confusion)
        # OA
        oa = accuracy_score(y_test, y_pred_test)
        # Kappa
        kappa = cohen_kappa_score(y_test, y_pred_test)

        confusion2 = confusion_matrix(y_true_valid, y_pred_valid)
        each_acc2, aa2 = AA_andEachClassAccuracy(confusion2)
        oa2 = accuracy_score(y_true_valid, y_pred_valid)
        kappa2 = cohen_kappa_score(y_true_valid, y_pred_valid)

        # 打印分类报告
        print("seed: ", seed)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print('Finished')
        print("\n======================================= Patch-wise Inference =======================================")
        # print('Patch-wise Inference (overall time): %.4fs' % (Patchwise_Inference_allTime))
        print(f'Patch-wise Inference (forward computation time): {Patchwise_Inference_forwardTime:.4f}s')
        print('Patch-wise Inference OA: ', oa)
        # print(classification)
        # print(confusion)
        print("Patch-wise Inference AA: ", aa)
        print("Patch-wise Inference Kappa: ", kappa)
        print("\n======================================= Full-image Inference =======================================")
        print('Full-image Inference (overall time): %.4fs' % (Fullimage_Inference_allTime))
        print(f'Full-image Inference (forward computation time): {Fullimage_Inference_forwardTime:.4f}s')
        print('Full-image Inference OA: ', oa2)
        print("Full-image Inference AA: ", aa2)
        print("Full-image Inference Kappa: ", kappa2)

        # 存储结果
        OA = oa
        AA = aa
        acc_class = each_acc
        OA_ALL.append(OA)
        AA_ALL.append(AA)
        KPP_ALL.append(kappa)
        AVG_ALL.append(acc_class)
        OA2 = oa2
        AA2 = aa2
        acc_class2 = each_acc2
        OA_ALL2.append(OA2)
        AA_ALL2.append(AA2)
        KPP_ALL2.append(kappa2)
        AVG_ALL2.append(acc_class2)
        Train_Time_ALL.append(Training_Time)
        Patchwise_Inference_Time_ALL.append(Patchwise_Inference_allTime)
        Patchwise_Inference_forwardTime_ALL.append(Patchwise_Inference_forwardTime)
        Fullimage_Inference_Time_ALL.append(Fullimage_Inference_allTime)
        Fullimage_Inference_forwardTime_ALL.append(Fullimage_Inference_forwardTime)

        save_root = './Results/{}/{}/num{}/iter{}'.format(net_name, args.data_name, args.n_validate, seed)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        # Draw_Classification_Map(classification_map, "./Results/" + dataset_name + str(OA))

        sio.savemat(save_root + '/training_time.mat', {'training_time': Training_Time})
        sio.savemat(save_root + '/Patchwise_Inference_classification_map.mat', {'Patchwise_Inference_classification_map': classification_map})
        Patchwise_Inference_CLSmetrics = np.append(acc_class, [OA, AA, kappa])
        sio.savemat(save_root + '/Patchwise_Inference_CLSmetrics.mat', {'Patchwise_Inference_CLSmetrics': Patchwise_Inference_CLSmetrics})
        sio.savemat(save_root + '/Patchwise_Inference_allTime.mat', {'Patchwise_Inference_allTime': Patchwise_Inference_allTime})
        sio.savemat(save_root + '/Patchwise_Inference_forwardTime.mat', {'Patchwise_Inference_forwardTime': Patchwise_Inference_forwardTime})

        Fullimage_Inference_CLSmetrics = np.append(acc_class2, [OA2, AA2, kappa2])
        sio.savemat(save_root + '/Fullimage_Inference_CLSmetrics.mat', {'Fullimage_Inference_CLSmetrics': Fullimage_Inference_CLSmetrics})
        sio.savemat(save_root + '/Fullimage_Inference_allTime.mat', {'Fullimage_Inference_allTime': Fullimage_Inference_allTime})
        sio.savemat(save_root + '/Fullimage_Inference_forwardTime.mat', {'Fullimage_Inference_forwardTime': Fullimage_Inference_forwardTime})

        # 记录到txt文件
        filename = f"/{net_name}_{args.data_name}_{seed}.txt"
        with open(save_root + filename, 'w', encoding='utf-8') as f:
            # 随机种子
            f.write('seed: %d\n' % seed)
            f.write(f"Total parameters: {total_params}\n")
            f.write(f"Trainable parameters: {trainable_params}\n")
            f.write(f"Finished\n")
            f.write(f"======================= Full-image Inference =======================\n")
            f.write('Full-image Inference (overall time): %.4fs\n' % (Fullimage_Inference_allTime))
            f.write(f'Full-image Inference (forward computation time): {Fullimage_Inference_forwardTime:.4f}s\n')
            f.write('Full-image inference AA: %.4f\n' % aa2)
            f.write('Full-image inference OA: %.4f\n' % oa2)
            f.write('Full-image inference Kappa: %.4f\n' % kappa2)
            f.write(f"======================= Patch-wise Inference =======================\n")
            # f.write('Patch-wise Inference (overall time): %.4fs\n' % (Patchwise_Inference_allTime))
            f.write(f'Patch-wise Inference (forward computation time): {Patchwise_Inference_forwardTime:.4f}s\n')
            f.write('Patch-wise Inference AA: %.4f\n' % aa)
            f.write('Patch-wise Inference OA: %.4f\n' % oa)
            f.write('Patch-wise Inference Kappa: %.4f\n' % kappa)
            f.write('each class accuracy:\n')
            for i, acc in enumerate(each_acc):
                f.write(f"{acc * 100:.2f}\n")
            f.write('classification report:\n')
            f.write(classification + '\n')
            f.write('confusion matrix:\n')
            f.write(str(confusion) + '\n')
        print(f"测试结果已保存到 {filename}")

        # 1. 删除大变量
        del outputs, out_cropped, y_pred_ALL2, inputs2
        # 2. 清理 CPU 内存
        gc.collect()
        # 3. 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Patchwise_Inference_Time_ALL = np.array(Patchwise_Inference_Time_ALL)
    Patchwise_Inference_forwardTime_ALL = np.array(Patchwise_Inference_forwardTime_ALL)

    OA_ALL2 = np.array(OA_ALL2)
    AA_ALL2 = np.array(AA_ALL2)
    KPP_ALL2 = np.array(KPP_ALL2)
    AVG_ALL2 = np.array(AVG_ALL2)
    Fullimage_Inference_Time_ALL = np.array(Fullimage_Inference_Time_ALL)
    Fullimage_Inference_forwardTime_ALL = np.array(Fullimage_Inference_forwardTime_ALL)

    print("\ntrain_ratio={}".format(args.n_validate))
    print("Average training time: {}".format(np.mean(Train_Time_ALL)))
    print("\n======================================= Patch-wise Inference =======================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average patch-wise inference (overall time): {}".format(np.mean(Patchwise_Inference_Time_ALL)))
    print("Average patch-wise inference (forward computation time): {}".format(np.mean(Patchwise_Inference_forwardTime_ALL)))
    print("\n======================================= Full-image Inference =======================================")
    print('OA=', np.mean(OA_ALL2), '+-', np.std(OA_ALL2))
    print('AA=', np.mean(AA_ALL2), '+-', np.std(AA_ALL2))
    print('Kpp=', np.mean(KPP_ALL2), '+-', np.std(KPP_ALL2))
    print('AVG=', np.mean(AVG_ALL2, 0), '+-', np.std(AVG_ALL2, 0))
    print("Average full-image inference (overall time): {}".format(np.mean(Fullimage_Inference_Time_ALL)))
    print("Average full-image inference (forward computation time): {}".format(np.mean(Fullimage_Inference_forwardTime_ALL)))

    # 保存数据信息
    f = open('./Results/' + f"{net_name}/" + args.data_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + "\ntrain_ratio={}".format(args.n_validate) \
                  + "\nAverage training time: {}".format(np.mean(Train_Time_ALL)) \
                  + "\n************************ Patch-wise Inference ************************" \
                  + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                  + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                  + "\nAverage patch-wise inference (overall time): {}".format(np.mean(Patchwise_Inference_Time_ALL)) \
                  + "\nAverage patch-wise inference (forward computation time): {}".format(np.mean(Patchwise_Inference_forwardTime_ALL)) \
                  + "\n************************ Full-image Inference ************************" \
                  + '\nOA=' + str(np.mean(OA_ALL2)) + '+-' + str(np.std(OA_ALL2)) \
                  + '\nAA=' + str(np.mean(AA_ALL2)) + '+-' + str(np.std(AA_ALL2)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL2)) + '+-' + str(np.std(KPP_ALL2)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL2, 0)) + '+-' + str(np.std(AVG_ALL2, 0)) \
                  + "\nAverage full-image inference (overall time): {}".format(np.mean(Fullimage_Inference_Time_ALL)) \
                  + "\nAverage full-image inference (forward computation time): {}".format(np.mean(Fullimage_Inference_forwardTime_ALL))
    f.write(str_results)
    f.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="data directory", default='./data')
    parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
    parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool,
                        default=False)
    args = parser.parse_args(args=[])

    main(args)

    # zpr_result_plot.draw()


def qtest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="data directory", default='./data')
    parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
    parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool,
                        default=False)
    args = parser.parse_args(args=[])
    txt = main(args)
    return txt
    # zpr_result_plot.draw()
