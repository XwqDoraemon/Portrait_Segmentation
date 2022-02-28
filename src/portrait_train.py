# -*- coding: utf-8 -*-
"""
# @file name  : portrait_train.py
# @author     : XueWQ
# @date       : 2021-06-12
# @brief      : 模型训练主代码
"""
import matplotlib
# matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torch.optim.lr_scheduler import LambdaLR
from datasets.portrait_dataset import PortraitDataset2000, PortraitDataset34427,PortraitDataset_fusion
from tools.model_trainer_bisenet import ModelTrainer
from tools.common_tools import setup_seed, show_confMat, plot_line, Logger, check_data_dir, create_logger
from config.portrait_config import cfg
from models.build_BiSeNet import BiSeNet
from tools.evalution_segmentaion import calc_semantic_segmentation_iou
from datetime import datetime
from losses.dice_loss import DiceLoss
from losses.focal_loss_binary import BinaryFocalLossWithLogits
from losses.bce_dice_loss import BCEDiceLoss ,FocDiceLoss
from tools.my_lr_schedule import CosineWarmupLr

setup_seed(12345)  # 先固定随机种子

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--lr', default=None, help='learning rate', type=float)
parser.add_argument('--max_epoch', default=None, type=int)
parser.add_argument('--train_bs', default=0, type=int)
parser.add_argument('--data_root_dir', default=r"G:\deep_learning_data\EG_dataset\dataset",
                    help="path to your dataset")
parser.add_argument('--ext_dir', default=r"G:\deep_learning_data\14w_matting",
                    help="path to your dataset")
parser.add_argument('--fusion_dir', default=r"G:\deep_learning_data\EG_dataset\data_aug_1700",
                    help="path to your dataset")
parser.add_argument('--output_dir', default=BASE_DIR,
                    help="path to your out_dir")
args = parser.parse_args()
cfg.lr_init = args.lr if args.lr else cfg.lr_init
cfg.train_bs = args.train_bs if args.train_bs else cfg.train_bs
cfg.max_epoch = args.max_epoch if args.max_epoch else cfg.max_epoch
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
if __name__ == '__main__':

    # 设置路径
    path_model_18 = os.path.join(BASE_DIR, "..", "..", "data", "pretrained_model", "resnet18-5c106cde.pth")  # bisenet
    path_model_101 = os.path.join(BASE_DIR, "..", "..", "preweights", "resnet101-5d3b4d8f.pth")  # bisenet
    
    # log 输出文件夹配置
    logger, log_dir = create_logger(args.output_dir)

    # ------------------------------------ step 1/5 : 加载数据------------------------------------
    # 构建Dataset实例
    root_dir = args.data_root_dir
    train_dir = os.path.join(root_dir, "training")
    valid_dir = os.path.join(root_dir, "testing")
    check_data_dir(train_dir)
    check_data_dir(valid_dir)

    train_set_list = []
    # base set
    # p_set_2000 = PortraitDataset2000(train_dir, in_size=cfg.in_size, transform=cfg.tf_train)
    p_set_2000 = PortraitDataset_fusion(train_dir,in_size= cfg.in_size,
                        transform=cfg.tf_train,coco_dir=cfg.coco_dir,
                        coco_datatype=cfg.coco_datatype)
    train_set_list.append(p_set_2000)

    if cfg.is_ext_data:
        p_set_34427 = PortraitDataset34427(args.ext_dir,  in_size=cfg.in_size, transform=cfg.tf_train, ext_num=cfg.ext_num)
        train_set_list.append(p_set_34427)
    if cfg.is_fusion_data:
        p_set_1700 = PortraitDataset2000(args.fusion_dir, in_size=cfg.in_size, transform=cfg.tf_train)
        train_set_list.append(p_set_1700)
    train_set = ConcatDataset(train_set_list)
    train_set.names = p_set_2000.names
    train_set.cls_num = p_set_2000.cls_num

    valid_set = PortraitDataset2000(valid_dir, in_size=cfg.in_size, transform=cfg.tf_valid)
    train_loader = DataLoader(train_set, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(valid_set, batch_size=cfg.valid_bs, num_workers=cfg.workers)

    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    cls = 1  # hardcode
    model = BiSeNet(cls, "resnet101", path_model_101)
    # model = BiSeNet(cls, "resnet18", path_model_18)
    model.to(cfg.device)

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
    if cfg.loss_type == "BCE":
        loss_f = nn.BCEWithLogitsLoss(pos_weight=cfg.bce_pos_weight)
    elif cfg.loss_type == "dice":
        loss_f = DiceLoss()
    elif cfg.loss_type == "BCE&dice":
        loss_f = BCEDiceLoss()
    elif cfg.loss_type == "focal":
        kwargs = {"alpha": cfg.focal_alpha, "gamma": cfg.focal_gamma, "reduction": 'mean'}
        loss_f = BinaryFocalLossWithLogits(**kwargs)
    elif cfg.loss_type == "focal&dice":
        kwargs = {"alpha": cfg.focal_alpha, "gamma": cfg.focal_gamma, "reduction": 'mean'}
        loss_f = FocDiceLoss(**kwargs)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    if cfg.is_warmup:
        # 注意，如果有这段warmup代码，一定要到trainer中修改 scheduler.step()
        iter_per_epoch = len(train_loader)
        scheduler = CosineWarmupLr(optimizer, batches=iter_per_epoch, max_epochs=cfg.max_epoch,
                                   base_lr=cfg.lr_init, final_lr=cfg.lr_final,
                                   warmup_epochs=cfg.warmup_epochs, warmup_init_lr=cfg.lr_warmup_init)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------
    # 记录训练配置信息
    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}"
                "GPU:{}".format(cfg, loss_f, scheduler, optimizer, model, torch.cuda.get_device_name()))

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    miou_rec = {"train": [], "valid": []}
    best_miou, best_epoch = 0, 0
    grad_lst_epoch = []
    for epoch in range(cfg.max_epoch):

        # 喂数据，训练模型
        loss_train, acc_train, mat_train, miou_train, grad_lst = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, scheduler, epoch, cfg, logger)
        loss_valid, acc_valid, mat_valid, miou_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, cfg)

        grad_lst_epoch.extend(grad_lst)

        # 学习率更新
        if not cfg.is_warmup:
            scheduler.step()

        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} LR:{} \n"
                    "Train loss:{:.4f} Train miou:{:.4f}\n"
                    "Valid loss:{:.4f} Valid miou:{:.4f}"
                    "". format(epoch, cfg.max_epoch, acc_train, acc_valid, optimizer.param_groups[0]["lr"],
                               loss_train, miou_train, loss_valid, miou_valid))

        # 记录训练信息
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        miou_rec["train"].append(miou_train), miou_rec["valid"].append(miou_valid)

        # 保存混淆矩阵图
        show_confMat(mat_train, train_set.names, "train", log_dir, epoch=epoch,
                     verbose=epoch == cfg.max_epoch - 1, perc=True)
        show_confMat(mat_valid, valid_set.names, "valid", log_dir, epoch=epoch,
                     verbose=epoch == cfg.max_epoch - 1, perc=True)
        # 保存loss曲线， acc曲线， miou曲线
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        plot_line(plt_x, miou_rec["train"], plt_x, miou_rec["valid"], mode="miou", out_dir=log_dir)
        # 保存模型
        if best_miou < miou_valid or epoch == cfg.max_epoch-1:

            best_epoch = epoch if best_miou < miou_valid else best_epoch
            best_miou = miou_valid if best_miou < miou_valid else best_miou
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_miou": best_miou}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == cfg.max_epoch-1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(args.output_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            # 观察各类别的iou：
            iou_array = calc_semantic_segmentation_iou(mat_valid)
            info = ["{}_iou:{:.4f}".format(n, iou) for n, iou in zip(train_set.names, iou_array)]
            logger.info("Best mIoU in {}. {}".format(epoch, "\t".join(info)))

    logger.info("{} done, best_miou: {:.4f} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_miou, best_epoch))

    if cfg.hist_grad:
        path_grad_png = os.path.join(log_dir, "grad_hist.png")
        logger.info("max grad in {}, is {}".format(grad_lst_epoch.index(max(grad_lst_epoch)), max(grad_lst_epoch)))
        import matplotlib.pyplot as plt
        plt.hist(grad_lst_epoch)
        plt.savefig(path_grad_png)
        logger.info(grad_lst_epoch)


