# -*- coding: utf-8 -*-
"""
# @file name  : camvid_config.py
# @author     : XueWQ
# @date       : 2021-03-12
# @brief      : dice loss
"""
import torch
import torch.nn as nn
from losses.dice_loss import DiceLoss
from losses.focal_loss_binary import BinaryFocalLossWithLogits

class BCEDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEDiceLoss, self).__init__()
        self.bce_func = nn.BCEWithLogitsLoss(**kwargs)  # *args和**kwargs，python可变长参数
        self.dice_func = DiceLoss()

    def forward(self, predict, target):
        loss_bce = self.bce_func(predict, target)
        loss_dice = self.dice_func(predict, target)
        return loss_dice + loss_bce
class FocDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(FocDiceLoss, self).__init__()
        self.foc_func = BinaryFocalLossWithLogits(**kwargs)
        self.dice_func = DiceLoss()

    def forward(self, predict, target):
        loss_bce = self.foc_func(predict, target)
        loss_dice = self.dice_func(predict, target)
        return loss_dice + loss_bce

if __name__ == "__main__":

    fake_out = torch.tensor([1, 1, -1, -1], dtype=torch.float32)
    fake_label = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    loss_f = BCEDiceLoss()
    loss = loss_f(fake_out, fake_label)

    print(loss)




