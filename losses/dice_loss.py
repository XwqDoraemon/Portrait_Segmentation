# -*- coding: utf-8 -*-
"""
# @file name  : camvid_config.py
# @author     : XueWQ
# @date       : 2021-03-12
# @brief      : dice loss
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    soft dice loss, 直接使用预测概率而不是使用阈值或将它们转换为二进制mask
    """
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        # pred不需要转bool变量，如https://github.com/yassouali/pytorch-segmentation/blob/master/utils/losses.py#L44
        # soft dice loss, 直接使用预测概率而不是使用阈值或将它们转换为二进制mask
        pred = torch.sigmoid(predict).view(num, -1)
        targ = target.view(num, -1)

        intersection = (pred * targ).sum()  # 利用预测值与标签相乘当作交集
        union = (pred + targ).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


if __name__ == "__main__":

    fake_out = torch.tensor([7, 7, -5, -5], dtype=torch.float32)
    fake_label = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    loss_f = DiceLoss()
    loss = loss_f(fake_out, fake_label)

    print(loss)




