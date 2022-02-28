# -*- coding: utf-8 -*-
"""
# @file name  : lr_range_test_segresnet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2021-03-12
# @brief      : lr 区间测试脚本
"""
import matplotlib
# matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.camvid_dataset import CamvidDataset
from tools.common_tools import setup_seed, show_confMat, plot_line, Logger, check_data_dir
from config.camvid_config import cfg
from models.deeplabv3_plus import DeepLabV3Plus
from models.unet import UNet, UNetResnet
from models.segnet import SegNet, SegResNet
from torch_lr_finder import LRFinder

setup_seed(12345)  # 先固定随机种子

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, help='learning rate', type=float)
parser.add_argument('--max_epoch', default=None)
parser.add_argument('--train_bs', default=0, type=int)
parser.add_argument('--data_root_dir', default=r"G:\deep_learning_data\camvid_from_paper",
                    help="path to your dataset")
args = parser.parse_args()
cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.train_bs if args.train_bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch

if __name__ == '__main__':
    # 设置路径
    path_model_50 = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "resnet50-19c8e357.pth")  # segnet
    path_model_101 = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "resnet101s-03a0f310.pth")  # deeplab
    path_model_50s = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "resnet50s-a75c83cf.pth")  # unet
    path_model_vgg = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "vgg16_bn-6c64b313.pth")
    # ------------------------------------ step 1/5 : 加载数据------------------------------------
    # 构建Dataset实例
    root_dir = args.data_root_dir
    train_img_dir = os.path.join(root_dir, 'train')
    train_lable_dir = os.path.join(root_dir, 'train_labels')
    path_to_dict = os.path.join(root_dir, 'class_dict.csv')
    check_data_dir(train_img_dir)
    check_data_dir(train_lable_dir)

    train_set = CamvidDataset(train_img_dir, train_lable_dir, path_to_dict, cfg.crop_size)

    train_loader = DataLoader(train_set, batch_size=cfg.train_bs, shuffle=True, num_workers=1)

    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    model = SegNet(num_classes=train_set.cls_num, path_model=path_model_vgg)
    # model = SegResNet(num_classes=train_set.cls_num, path_model=None)
    # model = UNet(num_classes=train_set.cls_num)
    # model = DeepLabV3Plus(num_classes=train_set.cls_num, path_model=path_model)
    model.to(cfg.device)

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
    loss_f = nn.CrossEntropyLoss().to(cfg.device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------
    # 接口：data loader、model、optimizer、loss_f
    lr_finder = LRFinder(model, optimizer, loss_f, device=cfg.device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph



