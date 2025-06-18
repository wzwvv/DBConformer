# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : alg_utils.py
import numpy as np
import torch
import torch.nn.functional as F

from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score


def EA(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def EA_online(x, R, sample_num):
    """
    Parameters
    ----------
    x : numpy array
        sample of shape (num_channels, num_time_samples)
    R : numpy array
        current reference matrix (num_channels, num_channels)
    sample_num: int
        previous number of samples used to calculate R

    Returns
    ----------
    refEA : numpy array
        data of shape (num_channels, num_channels)
    """

    cov = np.cov(x)
    refEA = (R * sample_num + cov) / (sample_num + 1)
    return refEA


def predict_pseudo_center(model, loader, args):
    """
    predict target pseudo labels with model and find class centers using pseudo prediction labels

    Parameters
    ----------
    model: torch.nn.module, EEGNet
    data_iter: torch.utils.data.DataLoader
    args: argparse.Namespace, for transfer learning

    Returns
    -------
    target_centers: torch tensors, cpu
    """
    start_test = True
    model.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_inputs = inputs.cpu()
                all_output = outputs.cpu()
                #all_label = labels.float().cpu()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.cpu()), 0)
                all_output = torch.cat((all_output, outputs.cpu()), 0)
                #all_label = torch.cat((all_label, labels.float().cpu()), 0)
    all_output = torch.nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    #true = all_label.cpu()
    #acc = accuracy_score(true, pred)
    for cls_id in range(args.class_num):
        indices = torch.where(pred == cls_id)[0]
        #print('cls_id:', cls_id)
        #print('indices:', indices)
        cls_data = torch.index_select(all_inputs, dim=0, index=indices)
        #print('cls_data.shape:', cls_data.shape)
        cls_center = torch.mean(cls_data, dim=0)
        #print('cls_center.shape:', cls_center.shape)
        if cls_id != 0:
            target_centers = torch.cat((cls_center, target_centers.float().cpu()), 0)
        else:
            target_centers = cls_center
    #print('target_centers.shape:', target_centers.shape)
    return target_centers

def SDR(source_iter, target_centers, mixup_ratio, args):
    """
    Source Domain Reconstruction
    https://ieeexplore.ieee.org/abstract/document/10216316/

    Parameters
    ----------
    source_iter: torch.utils.data.DataLoader
    target_centers: torch tensors, cpu
    mixup_ratio: float, hyperparameter for SDR
    args: argparse.Namespace, for transfer learning

    Returns
    -------
    source_recon_loader: torch.utils.data.DataLoader, cpu
    """
    start_test = True
    len_source = len(source_iter)
    with torch.no_grad():
        iter_source = iter(source_iter)
        for i in range(len(source_iter)):
            data = next(iter_source)
            inputs = data[0]
            labels = data[1]
            if start_test:
                all_inputs = inputs.cpu()
                all_label = labels.cpu()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.cpu()), 0)
                all_label = torch.cat((all_label, labels.cpu()), 0)

    for cls_id in range(args.class_num):
        indices = torch.where(all_label == cls_id)[0]
        #print('cls_id:', cls_id)
        #print('indices:', indices)
        cls_data = torch.index_select(all_inputs, dim=0, index=indices)
        #print('cls_data.shape:', cls_data.shape)
        cls_recon_data = cls_data * mixup_ratio + target_centers[cls_id] * (1 - mixup_ratio)
        cls_labels = torch.ones(len(indices)).reshape(-1, ) * cls_id
        #print('cls_recon_data.shape:', cls_recon_data.shape)
        if cls_id != 0:
            source_recon_data = torch.cat((source_recon_data, cls_recon_data.float().cpu()), 0)
            source_recon_label = torch.cat((source_recon_label, cls_labels.float()), 0)
        else:
            source_recon_data = cls_recon_data
            source_recon_label = cls_labels

    source_recon_data = source_recon_data.to(torch.float32)
    source_recon_label = source_recon_label.to(torch.long)
    if args.data_env != 'local':
        source_recon_data, source_recon_label = source_recon_data.cuda(), source_recon_label.cuda()

    dataset = torch.utils.data.TensorDataset(source_recon_data, source_recon_label)

    source_recon_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    assert len(source_recon_loader) == len_source, 'reconstructed data len not match!'

    return source_recon_loader
