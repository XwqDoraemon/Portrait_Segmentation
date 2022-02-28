# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 模型训练类
"""
import torch
import numpy as np
from collections import Counter
from tools.evalution_segmentaion import eval_semantic_segmentation
from torch.nn.utils import clip_grad_value_


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, cfg, optimizer, epoch_idx, logger):
        model.train()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        train_acc = []
        train_miou = []
        grad_lst_iter = []
        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            # forward & backward
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()

            if cfg.is_clip:
                clip_grad_value_(model.parameters(), cfg.clip_value)

            if cfg.hist_grad:
                grad_lst_tmp = []
                for p in model.parameters():
                    grad_lst_tmp.append(torch.max(p.grad.abs()).cpu().numpy())
                grad_lst_iter.append(max(grad_lst_tmp).flatten()[0])

            optimizer.step()

            # 评估
            pre_label = outputs.max(dim=1)[1].data.cpu().numpy()  # (bs, 360, 480)
            pre_label = [i for i in pre_label]  # 一个元素是一个样本的预测。pre_label[0].shape = (360,480)
            true_label = labels.data.cpu().numpy()
            true_label = [i for i in true_label]  # true_label[0].shape (360, 480)

            eval_metrix = eval_semantic_segmentation(pre_label, true_label, class_num)
            train_acc.append(eval_metrix['mean_class_accuracy'])
            train_miou.append(eval_metrix['miou'])
            conf_mat += eval_metrix["conf_mat"]
            loss_sigma.append(loss.item())

            # 间隔 log_interval 个iteration 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info('|Epoch[{}/{}]||batch[{}/{}]|batch_loss: {:.4f}||mIoU {:.4f}|'.format(
                    epoch_idx, cfg.max_epoch, i + 1, len(data_loader), loss.item(), eval_metrix['miou']))

        loss_mean = np.mean(loss_sigma)
        acc_mean = np.mean(train_acc)
        miou_mean = np.mean(train_miou)
        return loss_mean, acc_mean, conf_mat, miou_mean, grad_lst_iter

    @staticmethod
    def valid(data_loader, model, loss_f, cfg):
        model.eval()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        valid_acc = []
        valid_miou = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())

            # 统计loss
            loss_sigma.append(loss.item())

            # 评估
            pre_label = outputs.max(dim=1)[1].data.cpu().numpy()  # (bs, 360, 480)
            pre_label = [i for i in pre_label]  # 一个元素是一个样本的预测。pre_label[0].shape = (360,480)
            true_label = labels.data.cpu().numpy()
            true_label = [i for i in true_label]  # true_label[0].shape (360, 480)

            eval_metrix = eval_semantic_segmentation(pre_label, true_label, class_num)
            valid_acc.append(eval_metrix['mean_class_accuracy'])
            valid_miou.append(eval_metrix['miou'])
            conf_mat += eval_metrix["conf_mat"]
            loss_sigma.append(loss.item())

        loss_mean = np.mean(loss_sigma)
        acc_mean = np.mean(valid_acc)
        miou_mean = np.mean(valid_miou)

        return loss_mean, acc_mean, conf_mat, miou_mean


