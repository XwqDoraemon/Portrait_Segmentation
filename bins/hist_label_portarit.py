# -*- coding: utf-8 -*-
"""
# @file name  : hist_label_portrait.py
# @author     : https://github.com/TingsongYu
# @date       : 2020-03-11
# @brief      : 统计各类别数量
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
import cv2


def cal_cls_nums(path, t=0.78):
    label_img = cv2.imread(path)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    label_img[label_img > t] = 1
    label_img[label_img <= t] = 0

    label_img = label_img.flatten()
    count = np.bincount(label_img, minlength=2)  # np.bincount
    return count


if __name__ == '__main__':

    data_dir = r"G:\deep_learning_data\EG_dataset\dataset\training"

    counter = np.zeros((2,))
    # 遍历每张标签图，统计标签
    file_names = [n for n in os.listdir(data_dir) if n.endswith('_matte.png')]
    for i, name in enumerate(file_names):
        path_img = os.path.join(data_dir, name)
        counter += cal_cls_nums(path_img)   # 统计的数据记录于 counter中

    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=pos_weight
    # pos_weight设置为 负样本数量/正样本数量
    print(counter, counter[0] / counter[1])


