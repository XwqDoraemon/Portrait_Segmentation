# -*- coding: utf-8 -*-
"""
# @file name  : portrait_config.py
# @author     : XueWQ
# @date       : 2021-08
# @brief      : portrait 分割参数配置
"""
import torch
import torchvision.transforms as transforms
from easydict import EasyDict
import albumentations as A
import cv2

cfg = EasyDict()  # 访问属性的方式去使用key-value 即通过 .key获得value

cfg.is_fusion_data = False
cfg.is_ext_data = True
cfg.ext_num = 1700

# cfg.loss_type = "BCE"
cfg.loss_type = "BCE&dice"
# cfg.loss_type = "dice"
# cfg.loss_type = "focal"

cfg.focal_alpha = 0.5
cfg.focal_gamma = 2.  # 0.5， 2， 5， 10

# warmup cosine decay
cfg.is_warmup = True
cfg.warmup_epochs = 1
cfg.lr_final = 1e-5
cfg.lr_warmup_init = 0.  # 是0. 没错

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.max_epoch = 50  # 50

# batch size
cfg.train_bs = 8   # 32
cfg.valid_bs = 1 # 24
cfg.workers = 4  # 16

# 学习率
cfg.exp_lr = False   # 采用指数下降
cfg.lr_init = 0.01
cfg.factor = 0.1
cfg.milestones = [25, 45]
cfg.weight_decay = 5e-4
cfg.momentum = 0.9

cfg.log_interval = 10

cfg.bce_pos_weight = torch.tensor(0.75)  # 36/48 = 0.75  [36638126., 48661074.])   36/(36+48) = 0.42

cfg.in_size = 512   # 输入尺寸最短边

norm_mean = (0.5, 0.5, 0.5)  # 比imagenet的mean效果好
norm_std = (0.5, 0.5, 0.5)
# fusion
cfg.coco_dir = "/hy-tmp/coco"
cfg.coco_datatype = "val2017"
cfg.tf_train = A.Compose([
    A.Compose([
        A.Blur(blur_limit=7, p=0.5),  # 采用随机大小的kernel对图像进行模糊
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),  
        A.OneOf([
            A.IAAAffine(translate_percent=0.2, p=1),  # 平移
            A.IAAAffine(rotate=(-10, 10), p=1),  # 旋转 ["constant","edge","symmetric","reflect","wrap"]
        ], p=0.5)
    ]),
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])

cfg.tf_valid = A.Compose([
    A.Resize(width=cfg.in_size, height=cfg.in_size),
    A.Normalize(norm_mean, norm_std),
])
cfg.hist_grand = False




