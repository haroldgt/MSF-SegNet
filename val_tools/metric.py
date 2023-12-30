# -*- coding:utf-8 -*-
# author: Donglin zhu
# title: statistics and metric functions for all points classify for in a PointCloud
# @file: metric_.py

import numpy as np
import sys
import csv


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    # hist size: 19x19
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


def var_save_txt(var_save_path, var_ten_data):
    create_file = open(var_save_path, "w", encoding="utf-8")
    np.set_printoptions(threshold=sys.maxsize)
    create_file.write(str(var_ten_data))
    create_file.close()


def var_save_csv(var_save_path, var_ten_data):
    with open(var_save_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(var_ten_data)
