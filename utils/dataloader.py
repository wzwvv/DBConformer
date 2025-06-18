# -*- coding: utf-8 -*-
# @Time    : 2023/7/11
# @Author  : Siyang Li
# @File    : dataloader.py
import numpy as np
import torch
import mne
from sklearn import preprocessing
from sklearn.model_selection import KFold
from utils.data_utils import traintest_split_cross_subject, traintest_split_domain_classifier, domain_split_multisource, traintest_split_domain_classifier_pretest, traintest_split_cross_subject_trialdiffnum

from utils.data_utils import traintest_split_cross_subject_dwt

from utils.data_utils import traintest_split_cross_subject_dwt_class

from utils.data_utils import traintest_split_within_subject_Nsamples_diffnum, traintest_split_within_subject_Nsamples_samenum


def data_process(args, dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if 'BNCI2014001' in dataset:
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    elif dataset == 'MI1-7':
        X = np.load('./data/' + 'MI1' + '/X-7.npy')
        y = np.load('./data/' + 'MI1' + '/labels-7.npy')
    elif dataset == 'MI1':
        X = np.load('./data/' + 'MI1' + '/X-7.npy')
        y = np.load('./data/' + 'MI1' + '/labels-7.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    # print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22
        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:  # ['feet' 'left_hand' 'right_hand' 'tongue']
                indices.append(i)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22
        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'Zhou2016':
        paradigm = 'MI'
        num_subjects = 4
        sample_rate = 250
        ch_num = 14
        presum_trials_arr = np.array([[179, 150, 150],
                                      [150, 135, 150],
                                      [150, 151, 150],
                                      [135, 150, 150]])

        # only use session 1
        indices = []
        trials_arr = []
        for i in range(num_subjects):
            inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
            # only use two classes [right_hand, feet]
            cnt = 0
            for j in inds:
                if y[j] in ['left_hand', 'right_hand']:  # TODO left_hand, right_hand, feet
                    indices.append(j)
                    cnt += 1
            trials_arr.append(cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y = y.astype(int)
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
    elif dataset == 'Zhou2016_3':
        paradigm = 'MI'
        num_subjects = 4
        sample_rate = 250
        ch_num = 14

        presum_trials_arr = np.array([[179, 150, 150],
                                      [150, 135, 150],
                                      [150, 151, 150],
                                      [135, 150, 150]])

        # only use session 1
        indices = []
        trials_arr = []
        for i in range(num_subjects):
            inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
            # only use two classes [right_hand, feet]
            cnt = 0
            for j in inds:
                if y[j] in ['left_hand', 'right_hand', 'feet']:  # TODO left_hand, right_hand, feet
                    indices.append(j)
                    cnt += 1
            trials_arr.append(cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y[np.where(y == 'feet')] = 2
        y = y.astype(int)
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=10)  # TODO
        if args.backbone == 'shallow':
            print('downsample...')
            X = mne.filter.resample(X, down=4)  # TODO
    elif dataset == 'BNCI2014004':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

        trials_arr = np.array([[120, 120, 160, 160, 160],  # 720
                               [120, 120, 160, 120, 160],  # 680
                               [120, 120, 160, 160, 160],  # 720
                               [120, 140, 160, 160, 160],  # 740
                               [120, 140, 160, 160, 160],  # 740
                               [120, 120, 160, 160, 160],  # 720
                               [120, 120, 160, 160, 160],  # 720
                               [160, 120, 160, 160, 160],  # 760
                               [120, 120, 160, 160, 160]])  # 720

        # only use session 1 first 120 trials, remove other sessions TODO
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(120) + np.sum(trials_arr[:i, :]))  # TODO 120
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=10)  # TODO
        if args.backbone == 'shallow':
            print('downsample...')
            X = mne.filter.resample(X, down=4)  # TODO
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'MI1-7':
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 1000
        ch_num = 59
        print('MI downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)

        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
            sample_rate = int(sample_rate // 3)
    elif dataset == 'MI1':
        paradigm = 'MI'
        num_subjects = 5
        sample_rate = 1000
        ch_num = 59
        print('MI downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)
        X = np.concatenate([X[200:1000, :, :], X[1200:1400, :, :]], axis=0)  # S0 and S5 is left hand/right foot
        y = np.concatenate([y[200:1000], y[1200:1400]], axis=0)  # S0 and S5 is left hand/right foot
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
            sample_rate = int(sample_rate // 3)
    elif 'BNCI2014008' in dataset:
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    if dataset == 'Zhou2016':
        return X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr
    else:
        return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_process_secondsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session E, remove session T
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session test, remove session train
        indices = []
        for i in range(num_subjects):
            #indices.append(np.arange(100) + (160 * i))
            indices.append(np.arange(60) + (160 * i) + 100) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            # use second sessions
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7) + 200)
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7) + 200)
            else:
                indices.append(np.arange(200) + (400 * i) + 200)

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session E, remove session T
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288)
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def read_mi_combine_tar(args):

    if args.data_name == 'Zhou2016':
        X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_cross_subject_trialdiffnum(args.data, X, y, num_subjects, args.idt, trials_arr)
        args.trials_arr = trials_arr
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
        args.trials_arr = False

    return src_data, src_label, tar_data, tar_label

def read_mi_within_tar(args):

    if args.data_name == 'Zhou2016':
        X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_within_subject_Nsamples_diffnum(args, args.data, X, y, num_subjects, args.idt, args.nsamples, trials_arr)
        args.trials_arr = trials_arr
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_within_subject_Nsamples_samenum(args, X, y, num_subjects, args.idt, args.nsamples)
        args.trials_arr = False

    return src_data, src_label, tar_data, tar_label

def read_mi_within_tar_CV(args):
    if args.data_name == 'Zhou2016':
        X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
        X_t = data_subjects.pop(args.idt)
        y_t = labels_subjects.pop(args.idt)
        args.trials_arr = trials_arr
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
        X_t = data_subjects.pop(args.idt)
        y_t = labels_subjects.pop(args.idt)
        args.trials_arr = False
    return X_t, y_t

def read_mi_combine_tar_dwt(args):
    # if args.data_name == 'Zhou2016':
    #     X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
    #     src_data, src_label, tar_data, tar_label = traintest_split_cross_subject_trialdiffnum(args.data, X, y, num_subjects, args.idt, trials_arr)
    #     args.trials_arr = trials_arr
    # else:
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject_dwt_class(args, X, y, num_subjects, args.idt)
    args.trials_arr = False

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_tar_secondsession(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain_split(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier_pretest(args.data, X, y, num_subjects, args.ratio)

    return src_data, src_label, tar_data, tar_label


def read_mi_multi_source(args):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    data_subjects, labels_subjects = domain_split_multisource(args.data, X, y, num_subjects)

    return data_subjects, labels_subjects


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de
