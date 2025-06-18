'''
=================================================
coding:utf-8
@Time:      2025/4/25 20:42
@File:      DBConformer_CO.py
@Author:    Ziwei Wang
@Function:  Chronological Order (CO) scenario
=================================================
'''
import time
import math
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.network import backbone_net, backbone_net_deep, backbone_net_shallow, backbone_net_ifnet, \
    backbone_net_fbcnet, backbone_net_adfcnn, backbone_net_conformer, backbone_net_dbconformer, backbone_net_fbmsnet, \
    backbone_net_ifmambanet
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_within_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader, data_loader_within
import gc
import sys
import warnings
warnings.filterwarnings('ignore')


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_within_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader_within(X_src, y_src, X_tar, y_tar, args)
    # network selection
    if args.backbone == 'EEGNet':
        netF, netC = backbone_net(args, return_type='xy')
    elif args.backbone == 'deep':
        netF, netC = backbone_net_deep(args, return_type='xy')
    elif args.backbone == 'shallow':
        netF, netC = backbone_net_shallow(args, return_type='xy')
    elif args.backbone == 'IFNet':
        netF, netC = backbone_net_ifnet(args, return_type='xy')
    elif args.backbone == 'IFMambaNet':
        netF, netC = backbone_net_ifmambanet(args, return_type='xy')
    elif args.backbone == 'FBCNet':
        netF = backbone_net_fbcnet(args, return_type='xy')
    elif args.backbone == 'ADFCNN':
        netF, netC = backbone_net_adfcnn(args, return_type='xy')
    elif args.backbone == 'Conformer':
        netF = backbone_net_conformer(args, return_type='xy')
    elif args.backbone == 'DBConformer':
        netF = backbone_net_dbconformer(args)
    elif args.backbone == 'FBMSNet':
        netF, netC = backbone_net_fbmsnet(args, return_type='xy')
    if args.data_env != 'local':
        if args.backbone == 'FBCNet' or args.backbone == 'Conformer' or args.backbone == 'DBConformer':
            netF = netF.cuda()
            base_network = netF
            optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        else:
            netF, netC = netF.cuda(), netC.cuda()
            base_network = nn.Sequential(netF, netC)
            optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
            optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
    if args.class_num == 2:
        class_weight = torch.tensor([1., args.weight], dtype=torch.float32).cuda()  # class imbalance
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()
    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue
        iter_num += 1
        if 'ADFCNN' in args.backbone or 'Conformer' in args.backbone:
            inputs_source = inputs_source.unsqueeze_(3)
            inputs_source = inputs_source.permute(0, 3, 1, 2)
        features_source, outputs_source = base_network(inputs_source)
        classifier_loss = criterion(outputs_source, labels_source)
        optimizer_f.zero_grad()
        if args.backbone != 'FBCNet' and args.backbone != 'Conformer' and args.backbone != 'DBConformer':
            optimizer_c.zero_grad()
        classifier_loss.backward()
        optimizer_f.step()
        if args.backbone != 'FBCNet' and args.backbone != 'Conformer' and args.backbone != 'DBConformer':
            optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()
            acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)  # TODO target-online
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str,
                                                                   int(iter_num // len(dset_loaders["source"])),
                                                                   int(max_iter // len(dset_loaders["source"])),
                                                                   acc_t_te)
            args.log.record(log_str)
            base_network.train()
    print('Test Acc = {:.2f}%'.format(acc_t_te))
    print('saving model...')
    if not os.path.exists('./runs/' + str(args.data_name) + '/'):
        os.makedirs('./runs/' + str(args.data_name) + '/')
    if args.align:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
                       args.SEED) + '.ckpt')
    else:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(
                       args.SEED) + '_noEA' + '.ckpt')
    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()
    return acc_t_te


if __name__ == '__main__':
    cpu_num = 8
    torch.set_num_threads(cpu_num)
    data_name_list = ['Zhou2016']  # 'BNCI2014001', 'BNCI2014004', 'Zhou2016', 'MI1-7', 'BNCI2014002'
    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])
    for data_name in data_name_list:
        weight = 1
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 2, 1001, 250, 144
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 14, 15, 2, 2561, 512, 100
        if data_name == 'BNCI2014004': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 3, 2, 1126, 250, 120
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 12, 13, 2, 2561, 512, 200
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 4, 1001, 250, 288
        if data_name == 'MI1-7': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 7, 59, 2, 750, 250, 200
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, dim_e, dim_p = 'MI', 5, 59, 2, 750, 250, 200, 184, 750
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 8, 8, 2, 256, 256, 4200, 64,
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 10, 8, 2, 206, 256, 2520, 64
        if data_name == 'Zhou2016': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 2, 1251, 250, -1
        if data_name == 'Zhou2016_3': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 3, 1251, 250, -1
        F1, D, F2 = 4, 2, 8

        if 'BNCI2014008' in data_name:
            F1, D, F2 = 8, 4, 16
        args = argparse.Namespace(trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name,
                                  F1=F1, D=D, F2=F2, weight=weight)

        args.backbone = 'DBConformer'  # DBConformer (Ours)
        args.method = args.backbone + '_' + data_name
        # DBConformer parameters
        args.gate_flag = False  # reduce performance
        args.posemb_flag = True  # enhance performance
        args.chn_atten_flag = True  # depend on datasets
        args.branch = 'all'  # [all, temporal]
        if args.backbone == 'DBConformer':
            args.emb_size = 40
            args.spa_dim = 16
            if data_name == 'BNCI2014002':
                args.transformer_depth_tem = 4
                args.transformer_depth_chn = 4
            else:
                args.transformer_depth_tem = 2
                args.transformer_depth_chn = 2
            if data_name == 'BNCI2015001' or data_name == 'BNCI2014002':
                args.patch_size = 128
            else:
                args.patch_size = 125
        # whether to use EA
        args.align = True
        args.dropoutRate = 0.25
        # learning rate
        args.lr = 0.0001
        # batch size
        args.batch_size = 32
        # training epochs
        args.max_epoch = 200
        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'

        total_acc = []
        for s in [1, 2, 3, 4, 5]:
            args.SEED = s
            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True
            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)
            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)
            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log
                if args.data_name == 'Zhou2016':
                    sbj_num = [119, 100, 100, 90]
                    args.nsamples = math.ceil(sbj_num[idt] / 2 * 0.8)  # 80% training，20% testing
                elif args.data_name == 'BNCI2014001-4':
                    args.nsamples = math.ceil(args.trial_num / 4 * 0.8)
                else:
                    args.nsamples = math.ceil(args.trial_num / 2 * 0.8)
                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)
            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)
        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)
        print(str(total_acc))
        args.log.record(str(total_acc))
        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)
        print(subject_mean)
        print(args.method)
        print(total_mean)
        print(total_std)
        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))
