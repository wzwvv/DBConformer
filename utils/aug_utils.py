# -*- coding: utf-8 -*-
# @Time    : 2023/07/17
# @Author  : Siyang Li
# @File    : aug_utils.py
import einops
import torch
import numpy as np
import random


def random_upsampling_transform(X, ratio=None):
    """

    Parameters
    ----------
    X: torch tensor of shape (num_samples, num_channels, num_timesamples)
    ratio: float between (0.0, 0.5)

    Returns
    -------
    transformedX: upsampled signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """
    revert_shape = False
    if len(X.shape) == 4:
        revert_shape = True
        X = torch.squeeze(X, 1)
    if len(X.shape) == 2:
        X = X.unsqueeze_(0)
    if len(X.shape) != 3:
        return
    num_samples, num_channels, num_timesamples = X.shape
    num_throw = int(num_timesamples * ratio)
    num_kept = num_timesamples - num_throw
    transformedX = []
    for sample in X:
        cut = sample[:, num_throw // 2 - 1:-(num_throw - num_throw // 2)]
        start = random.randint(0, num_kept - num_throw - 1)
        before_stretch = cut[:, start:start + num_throw]
        avg_arr = ((before_stretch + torch.roll(before_stretch, -1)) / 2.0)
        after_stretch = []
        for i in range(len(before_stretch)):
            a = torch.flatten(torch.vstack([before_stretch[i], avg_arr[i]]).permute((1, 0)))[:-1]
            after_stretch.append(a)
        after_stretch = torch.stack(after_stretch)
        transformed_sample = torch.cat((cut[:, :start], after_stretch, cut[:, start + num_throw:]), dim=1)
        transformedX.append(transformed_sample)
    transformedX = torch.stack(transformedX)
    if revert_shape:
        transformedX = transformedX.unsqueeze_(1)
    return transformedX


def leftrightflipping_transform(X, left_mat, right_mat):
    """

    Parameters
    ----------
    X: torch tensor of shape (num_samples, 1, num_channels, num_timesamples)
    left_mat: numpy array of shape (a, ), where a is the number of left brain channels, in order
    right_mat: numpy array of shape (b, ), where b is the number of right brain channels, in order

    Returns
    -------
    transformedX: transformed signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """

    num_samples, _, num_channels, num_timesamples = X.shape
    transformedX = torch.zeros((num_samples, 1, num_channels, num_timesamples))
    for ch in range(num_channels):
        if ch in left_mat:
            ind = left_mat.index(ch)
            transformedX[:, 0, ch, :] = X[:, 0, right_mat[ind], :]
        elif ch in right_mat:
            ind = right_mat.index(ch)
            transformedX[:, 0, ch, :] = X[:, 0, left_mat[ind], :]
        else:
            transformedX[:, 0, ch, :] = X[:, 0, ch, :]

    return transformedX


def leftrightdecay_transform(X, y, left_mat, right_mat, dataset, ratio=0.8):
    """

    Parameters
    ----------
    X: torch tensor of shape (num_samples, 1, num_channels, num_timesamples)
    left_mat: numpy array of shape (a, ), where a is the number of left brain channels, in order
    right_mat: numpy array of shape (b, ), where b is the number of right brain channels, in order

    Returns
    -------
    transformedX: transformed signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """

    num_samples, _, num_channels, num_timesamples = X.shape

    if dataset == 'BNCI2014002' or dataset == 'BNCI2015001':
        # assume binary
        class_0 = X[torch.where(y == 0)[0]]
        class_0[:, :, right_mat, :] *= ratio

        class_1 = X[torch.where(y == 1)[0]]
    else:
        # assume binary
        class_0 = X[torch.where(y == 0)[0]]
        class_0[:, :, left_mat, :] *= ratio

        class_1 = X[torch.where(y == 1)[0]]
        class_1[:, :, right_mat, :] *= ratio

    transformedX = torch.concat((class_0, class_1))

    return transformedX


def leftrightadd_transform(X, y, left_mat, right_mat, dataset, ratio=0.8):
    """

    Parameters
    ----------
    X: torch tensor of shape (num_samples, 1, num_channels, num_timesamples)
    left_mat: numpy array of shape (a, ), where a is the number of left brain channels, in order
    right_mat: numpy array of shape (b, ), where b is the number of right brain channels, in order

    Returns
    -------
    transformedX: transformed signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """

    num_samples, _, num_channels, num_timesamples = X.shape

    if dataset == 'BNCI2014002' or dataset == 'BNCI2015001':
        # assume binary
        class_0 = X[torch.where(y == 0)[0]]
        class_0[:, :, right_mat, :] *= ratio
        class_1 = X[torch.where(y == 1)[0]]
    elif dataset == 'BNCI2014001' or dataset == 'BNCI2014004':
        # assume binary
        class_0 = X[torch.where(y == 0)[0]]
        class_0[:, :, left_mat, :] *= ratio
        class_1 = X[torch.where(y == 1)[0]]
        class_1[:, :, right_mat, :] *= ratio
    elif dataset == 'BNCI2014008':
        class_0 = X[torch.where(y == 0)[0]]
        class_1 = X[torch.where(y == 1)[0]]
        k = torch.std_mean(class_0, dim=0, keepdim=True)
        mean_cls0 = einops.repeat(k[0], 'h w j m -> (repeat h) w j m', repeat=class_1.shape[0])
        std_cls0 = einops.repeat(k[1], 'h w j m -> (repeat h) w j m', repeat=class_1.shape[0])
        class_1_add = class_1 + mean_cls0
        class_1_sub = class_1 - mean_cls0
    transformedX = torch.concat((class_1_add, class_1_sub))
    return transformedX


def small_laplace_normalize(X, adj_mat):
    """

    Parameters
    ----------
    X: numpy array of shape (num_samples, num_channels, num_timesamples)
    adj_mat: numpy array of shape (num_connections, 2)

    Returns
    -------
    transformedX: transformed signal of numpy array of shape (num_samples, num_channels, num_timesamples)
    """
    num_samples, num_channels, num_timesamples = X.shape
    transformedX = np.zeros((num_samples, num_channels, num_timesamples))
    for ch in range(num_channels):
        adj_chs = []
        for i in range(len(adj_mat)):
            a, b = adj_mat[i]
            if ch == a:
                adj_chs.append(b)
            elif ch == b:
                adj_chs.append(a)
        transformedX[:, ch, :] = np.average(X[:, adj_chs, :], axis=1)
    return transformedX