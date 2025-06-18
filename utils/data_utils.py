# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : data_utils.py
import random
import pywt
import numpy as np

from utils.alg_utils import EA


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    print(label_01)
    return label_01


def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


def traintest_split_cross_subject(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y

def traintest_split_cross_subject_dwt(args, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    l_id = [i for i in range(train_y.shape[0]) if train_y[i] == 0]
    r_id = [i for i in range(train_y.shape[0]) if train_y[i] == 1]
    Xs0 = train_x[l_id]
    ys0 = train_y[l_id]
    Xs1 = train_x[r_id]
    ys1 = train_y[r_id]

    l_index = [i for i in range(test_y.shape[0]) if test_y[i] == 0]
    r_index = [i for i in range(test_y.shape[0]) if test_y[i] == 1]
    X_la = test_x[l_index, :, :]
    X_ra = test_x[r_index, :, :]
    y_la = test_y[l_index]
    y_ra = test_y[r_index]
    nsamples = 1
    # target train data
    X_lat = X_la[:nsamples, :, :]
    X_rat = X_ra[:nsamples, :, :]
    y_lat = y_la[:nsamples]
    y_rat = y_ra[:nsamples]
    # target test data
    X_lae = X_la[nsamples:, :, :]
    X_rae = X_ra[nsamples:, :, :]
    y_lae = y_la[nsamples:]
    y_rae = y_ra[nsamples:]
    X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
    y_tar_e = np.concatenate((y_lae, y_rae), axis=0)

    if args.DWT_aug:
        src_aug_x = []
        src_aug_y = []
        wavename = 'db5'  # TODO db5
        TcA0, TcD0 = pywt.dwt(X_lat, wavename)
        TcA1, TcD1 = pywt.dwt(X_rat, wavename)
        for k in range(Xs0.shape[0]):
            Xss0 = Xs0[k, :, :]
            Xss0 = np.expand_dims(Xss0, axis=0)
            ScA0, ScD0 = pywt.dwt(Xss0, wavename)  # TODO dwt
            Xs_aug0 = pywt.idwt(ScA0, TcD0, 'db5', 'smooth')
            Xt_aug0 = pywt.idwt(TcA0, ScD0, 'db5', 'smooth')
            # if args.align:
            #     Xs_aug0 = EA(Xs_aug0)
            #     Xt_aug0 = EA(Xt_aug0)
            src_aug_x.append(Xs_aug0[:, :, :Xss0.shape[-1]])
            src_aug_x.append(Xt_aug0[:, :, :Xss0.shape[-1]])
            src_aug_y.append(y_lat)
            src_aug_y.append(y_lat)
        for m in range(Xs1.shape[0]):
            Xss1 = Xs1[m, :, :]
            Xss1 = np.expand_dims(Xss1, axis=0)
            ScA1, ScD1 = pywt.dwt(Xss1, wavename)  # TODO dwt
            Xs_aug1 = pywt.idwt(ScA1, TcD1, 'db5', 'smooth')
            Xt_aug1 = pywt.idwt(TcA1, ScD1, 'db5', 'smooth')
            # if args.align:
            #     Xs_aug1 = EA(Xs_aug1)
            #     Xt_aug1 = EA(Xt_aug1)
            src_aug_x.append(Xs_aug1[:, :, :Xss1.shape[-1]])
            src_aug_x.append(Xt_aug1[:, :, :Xss1.shape[-1]])
            src_aug_y.append(y_rat)
            src_aug_y.append(y_rat)
        src_aug_x = np.concatenate(src_aug_x)
        src_aug_y = np.concatenate(src_aug_y)
        src_x = np.concatenate((Xs0, Xs0, src_aug_x))
        src_y = np.concatenate((ys0, ys1, src_aug_y))
        return src_x, src_y, test_x, test_y


def traintest_split_cross_subject_dwt_class(args, X, y, num_subjects, test_subject_id):
    src_x = []
    src_y = []
    src_x_0 = []
    src_y_0 = []
    src_x_1 = []
    src_y_1 = []
    for k in range(num_subjects):
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
        if k != test_subject_id:
            X_s = data_subjects[k]
            y_s = labels_subjects[k]
            if args.align:
                X_s = EA(X_s)
            l_id = [i for i in range(y_s.shape[0]) if y_s[i] == 0]
            r_id = [i for i in range(y_s.shape[0]) if y_s[i] == 1]
            Xs0 = X_s[l_id]
            ys0 = y_s[l_id]
            Xs1 = X_s[r_id]
            ys1 = y_s[r_id]
            src_x.append(X_s)
            src_y.append(y_s)
            src_x_0.append(Xs0)
            src_y_0.append(ys0)
            src_x_1.append(Xs1)
            src_y_1.append(ys1)
            # print(len(src_x_0), Xs0.shape, len(src_x_1), Xs1.shape)
        else:
            nsamples = 1
            X_t = data_subjects.pop(test_subject_id)
            y_t = labels_subjects.pop(test_subject_id)
            if args.align:
                X_t = EA(X_t)
            l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
            r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
            X_la = X_t[l_index, :, :]
            X_ra = X_t[r_index, :, :]
            y_la = y_t[l_index]
            y_ra = y_t[r_index]
            # target train data
            X_lat = X_la[:nsamples, :, :]
            X_rat = X_ra[:nsamples, :, :]
            y_lat = y_la[:nsamples]
            y_rat = y_ra[:nsamples]
            # target test data
            X_lae = X_la[nsamples:, :, :]
            X_rae = X_ra[nsamples:, :, :]
            y_lae = y_la[nsamples:]
            y_rae = y_ra[nsamples:]
            X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
            y_tar_e = np.concatenate((y_lae, y_rae), axis=0)
    if args.DWT_aug:
        src_aug_x = []
        src_aug_y = []
        wavename = 'db5'  # TODO db5
        TcA0, TcD0 = pywt.dwt(X_lat.squeeze(), wavename)
        TcA1, TcD1 = pywt.dwt(X_rat.squeeze(), wavename)
        for m in range(len(src_x_0)):
            Xs0 = src_x_0[m]
            for k in range(Xs0.shape[0]):
                Xss0 = Xs0[k, :, :]
                ScA0, ScD0 = pywt.dwt(Xss0, wavename)  # TODO dwt
                Xs_aug0 = pywt.idwt(ScA0, TcD0, 'db5', 'smooth')
                Xt_aug0 = pywt.idwt(TcA0, ScD0, 'db5', 'smooth')
                if args.align:
                    Xs_aug0 = EA(Xs_aug0)
                    Xt_aug0 = EA(Xt_aug0)
                src_aug_x.append(Xs_aug0[:, :Xss0.shape[-1]])
                src_aug_x.append(Xt_aug0[:, :Xss0.shape[-1]])
                src_aug_y.append(y_lat)
                src_aug_y.append(y_lat)
        for m in range(len(src_x_1)):
            Xs1 = src_x_1[m]
            for k in range(Xs1.shape[0]):
                Xss1 = Xs1[k]
                ScA1, ScD1 = pywt.dwt(Xss1, wavename)  # TODO dwt
                Xs_aug1 = pywt.idwt(ScA1, TcD1, 'db5', 'smooth')
                Xt_aug1 = pywt.idwt(TcA1, ScD1, 'db5', 'smooth')
                if args.align:
                    Xs_aug1 = EA(Xs_aug1)
                    Xt_aug1 = EA(Xt_aug1)
                src_aug_x.append(Xs_aug1[:, :Xss1.shape[-1]])
                src_aug_x.append(Xt_aug1[:, :Xss1.shape[-1]])
                src_aug_y.append(y_rat)
                src_aug_y.append(y_rat)
        src_aug_x = np.array(src_aug_x)
        src_aug_y = np.array(src_aug_y).squeeze()
        src_x = np.concatenate(src_x)
        src_y = np.concatenate(src_y)
        src_x = np.concatenate((src_x, src_aug_x))
        src_y = np.concatenate((src_y, src_aug_y))
        return src_x, src_y, X_t, y_t
    else:
        src_x = np.concatenate(src_x)
        src_y = np.concatenate(src_y)
        return src_x, src_y, X_t, y_t

def traintest_split_cross_subject_trialdiffnum(dataset, X, y, num_subjects, test_subject_id, trials_arr):
    if trials_arr:
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        print(accum_arr)
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
    else:
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_within_subject_Nsamples_diffnum(args, dataset, X, y, num_subjects, test_subject_id, nsamples, trials_arr):
    if 'Zhou2016' in dataset or 'HighGamma' in dataset or 'Weibo2014' in dataset:
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
    else:
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    X_t = data_subjects.pop(test_subject_id)
    y_t = labels_subjects.pop(test_subject_id)
    l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
    r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
    X_la = X_t[l_index, :, :]
    X_ra = X_t[r_index, :, :]
    y_la = y_t[l_index]
    y_ra = y_t[r_index]
    X_lat = X_la[:nsamples, :, :]
    X_rat = X_ra[:nsamples, :, :]
    y_lat = y_la[:nsamples]
    y_rat = y_ra[:nsamples]
    X_lae = X_la[nsamples:, :, :]
    X_rae = X_ra[nsamples:, :, :]
    y_lae = y_la[nsamples:]
    y_rae = y_ra[nsamples:]
    X_tar_t = np.concatenate((X_lat, X_rat), axis=0)
    y_tar_t = np.concatenate((y_lat, y_rat), axis=0)
    X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
    y_tar_e = np.concatenate((y_lae, y_rae), axis=0)
    print(X_tar_t.shape[0], X_tar_e.shape[0])
    if args.align:
        X_tar_t = EA(X_tar_t)
        X_tar_e = EA(X_tar_e)
    return X_tar_t, y_tar_t, X_tar_e, y_tar_e

def traintest_split_within_subject_Nsamples_samenum(args, X, y, num_subjects, test_subject_id, nsamples):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    X_t = data_subjects.pop(test_subject_id)
    y_t = labels_subjects.pop(test_subject_id)
    if args.data_name != 'BNCI2014001-4':
        l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
        r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
        X_la = X_t[l_index, :, :]
        X_ra = X_t[r_index, :, :]
        y_la = y_t[l_index]
        y_ra = y_t[r_index]
        X_lat = X_la[:nsamples, :, :]
        X_rat = X_ra[:nsamples, :, :]
        y_lat = y_la[:nsamples]
        y_rat = y_ra[:nsamples]
        X_lae = X_la[nsamples:, :, :]
        X_rae = X_ra[nsamples:, :, :]
        y_lae = y_la[nsamples:]
        y_rae = y_ra[nsamples:]
        X_tar_t = np.concatenate((X_lat, X_rat), axis=0)
        y_tar_t = np.concatenate((y_lat, y_rat), axis=0)
        X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
        y_tar_e = np.concatenate((y_lae, y_rae), axis=0)
    else:
        l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
        r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
        f_index = [i for i in range(y_t.shape[0]) if y_t[i] == 2]
        t_index = [i for i in range(y_t.shape[0]) if y_t[i] == 3]
        X_l = X_t[l_index, :, :]
        X_r = X_t[r_index, :, :]
        X_f = X_t[f_index, :, :]
        X_t = X_t[t_index, :, :]
        y_l = y_t[l_index]
        y_r = y_t[r_index]
        y_f = y_t[f_index]
        y_t = y_t[t_index]
        X_lat = X_l[:nsamples, :, :]
        X_rat = X_r[:nsamples, :, :]
        X_fat = X_f[:nsamples, :, :]
        X_tat = X_t[:nsamples, :, :]
        y_lat = y_l[:nsamples]
        y_rat = y_r[:nsamples]
        y_fat = y_f[:nsamples]
        y_tat = y_t[:nsamples]
        X_lae = X_l[nsamples:, :, :]
        X_rae = X_r[nsamples:, :, :]
        X_fae = X_f[nsamples:, :, :]
        X_tae = X_t[nsamples:, :, :]
        y_lae = y_l[nsamples:]
        y_rae = y_r[nsamples:]
        y_fae = y_f[nsamples:]
        y_tae = y_t[nsamples:]
        X_tar_t = np.concatenate((X_lat, X_rat, X_fat, X_tat), axis=0)
        y_tar_t = np.concatenate((y_lat, y_rat, y_fat, y_tat), axis=0)
        X_tar_e = np.concatenate((X_lae, X_rae, X_fae, X_tae), axis=0)
        y_tar_e = np.concatenate((y_lae, y_rae, y_fae, y_tae), axis=0)
    print(X_tar_t.shape[0], X_tar_e.shape[0])
    if args.align:
        X_tar_t = EA(X_tar_t)
        X_tar_e = EA(X_tar_e)
    return X_tar_t, y_tar_t, X_tar_e, y_tar_e


def traintest_split_domain_classifier(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    data_subjects.pop(test_subject_id)
    labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    for i in range(num_subjects - 1):
        labels_subjects[i] = np.ones((int(len(labels_subjects[i]))),) * i
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training:', train_x.shape, train_y.shape)
    return train_x, train_y, None, None


def traintest_split_domain_classifier_pretest(dataset, X, y, num_subjects, ratio):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    train_x_all = []
    train_y_all = []
    test_x_all = []
    test_y_all = []
    for i in range(num_subjects):
        data = data_subjects[i]
        random.shuffle(data)
        train_x_all.append(data[:int(len(data) * ratio)])
        train_y_all.append(np.ones((int(len(data) * ratio)),) * i)
        test_x_all.append(data[int(len(data) * ratio):])
        test_y_all.append(np.ones((int(len(data) * (1 - ratio))),) * i)
    train_x = np.concatenate(train_x_all, axis=0)
    train_y = np.concatenate(train_y_all, axis=0)
    test_x = np.concatenate(test_x_all, axis=0)
    test_y = np.concatenate(test_y_all, axis=0)
    print('Training/Test split:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


def domain_split_multisource(dataset, X, y, num_subjects):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    print(num_subjects, 'Subjects of', data_subjects[0].shape, labels_subjects[0].shape)
    return data_subjects, labels_subjects
