# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer_bisenet.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : bisenet模型训练类
"""
import torch
import numpy as np
from collections import Counter
from tools.evalution_segmentaion import eval_semantic_segmentation
from torch.nn.utils import clip_grad_value_


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, cfg, logger):
        model.train()

        class_num = data_loader.dataset.cls_num
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        train_acc = []
        train_miou = []
        train_class_acc = []
        grad_lst_iter = []
        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            # forward & backward
            outputs, output_sup1, output_sup2 = model(inputs)

            # compute loss
            labels_4d = labels.unsqueeze(1)    # b,h,w --> b,c,h,w

            loss_1 = loss_f(outputs, labels_4d)
            loss_2 = loss_f(output_sup1, labels_4d)
            loss_3 = loss_f(output_sup2, labels_4d)

            loss = loss_1 + loss_2 + loss_3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # warmup 每个iteration执行step
            if cfg.is_warmup:
                scheduler.step()

            # 评估 IoU
            # 预测，连续变量转binary
            if outputs.shape[1] == 1:
                outputs_prob = torch.sigmoid(outputs).squeeze(1)    # 转为概率形式 0-1之间
                outputs_b = outputs_prob > 0.78    # 转为binary形式，0. 是借鉴SINet
                outputs_b = outputs_b.long().data.cpu().numpy()
            else:
                outputs_b = outputs.max(dim=1)[1].data.cpu().numpy()  # (4, 352, 480)

            outputs_b = [i for i in outputs_b]      # pre_label[0].shape  (600, 600)
            # 标签，连续变量转binary
            labels_b = labels > 0.78
            labels_b = labels_b.type(torch.int).data.cpu().numpy()
            labels_b = [i for i in labels_b]  # 一个元素是一个样本

            eval_metrix = eval_semantic_segmentation(outputs_b, labels_b, class_num)
            train_acc.append(eval_metrix['mean_class_accuracy'])
            train_miou.append(eval_metrix['iou'][1])  # 只取前景类IoU
            train_class_acc.append(eval_metrix['class_accuracy'])
            conf_mat += eval_metrix["conf_mat"]
            loss_sigma.append(loss_1.item())  # 记录loss只看 主loss，以便于valid比对

            # 每10个iteration 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info('|Epoch[{}/{}]||batch[{}/{}]|batch_loss: {:.4f}||mIoU {:.4f}|'.format(
                    epoch_idx, cfg.max_epoch, i + 1, len(data_loader), loss_1.item(), eval_metrix['iou'][1]))

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
        valid_cls_acc = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            outputs = model(inputs)

            # loss
            labels_4d = labels.unsqueeze(1)    # b,h,w --> b,c,h,w
            loss = loss_f(outputs, labels_4d)
            loss_sigma.append(loss.item())      # 记录loss

            # 评估 IoU
            # 预测，连续变量转binary
            if outputs.shape[1] == 1:
                outputs_prob = torch.sigmoid(outputs).squeeze(1)    # 转为概率形式 0-1之间
                outputs_b = outputs_prob > 0.78    # 转为binary形式，0. 是借鉴SINet
                outputs_b = outputs_b.long().data.cpu().numpy()
            else:
                outputs_b = outputs.max(dim=1)[1].data.cpu().numpy()  # (4, 352, 480)
            outputs_b = [i for i in outputs_b]      # pre_label[0].shape  (600, 600)
            # 标签，连续变量转binary
            labels_b = labels > 0.78
            labels_b = labels_b.type(torch.int).data.cpu().numpy()
            labels_b = [i for i in labels_b]  # 一个元素是一个样本

            eval_metrix = eval_semantic_segmentation(outputs_b, labels_b, class_num)

            valid_acc.append(eval_metrix['mean_class_accuracy'])
            valid_miou.append(eval_metrix['iou'][1])  # 只取前景类IoU
            valid_cls_acc.append(eval_metrix['class_accuracy'])
            conf_mat += eval_metrix["conf_mat"]
            loss_sigma.append(loss.item())

        loss_mean = np.mean(loss_sigma)
        acc_mean = np.mean(valid_acc)
        miou_mean = np.mean(valid_miou)

        return loss_mean, acc_mean, conf_mat, miou_mean


