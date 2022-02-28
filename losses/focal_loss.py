# -*- coding: utf-8 -*-
"""
# @file name  : focal_loss.py
# @author     : XueWQ
# @date       : 2021-06-03
# @brief      : 标准的 focal loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)  # 因CE中取了log，所以要exp回来，就得到概率。因为输入并不是概率，CEloss中自带softmax转为概率形式
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


if __name__ == "__main__":

    target = torch.tensor([1], dtype=torch.long)
    gamma_lst = [0, 0.5, 1, 2, 5]
    loss_dict = {}
    for gamma in gamma_lst:
        focal_loss_func = FocalLoss(gamma=gamma)
        loss_dict.setdefault(gamma, [])

        for i in np.linspace(0.5, 10.0, num=30):
            outputs = torch.tensor([[5, i]], dtype=torch.float)  # 制造不同概率的输出
            prob = F.softmax(outputs, dim=1)  # 由于pytorch的CE自带softmax，因此想要知道具体预测概率，需要自己softmax
            loss = focal_loss_func(outputs, target)
            loss_dict[gamma].append((prob[0, 1].item(), loss.item()))


    for gamma, value in loss_dict.items():
        x_prob = [prob for prob, loss in value]
        y_loss = [loss for prob, loss in value]
        plt.plot(x_prob, y_loss, label="γ="+str(gamma))

    plt.title("Focal Loss")
    plt.xlabel("probability of ground truth class")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


