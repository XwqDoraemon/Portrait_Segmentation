# -*- coding: utf-8 -*-
"""
# @file name  : fusion_img.py
# @author     : hXueWQ
# @date       : 2020-05-28
# @brief      : 采用现有分割图像合成新分割数据集
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
import cv2
from pycocotools.coco import COCO
from functools import partial


class CocoImg(object):
    """
    根据指定类别挑选图片，并过滤掉图片中存在的类别
    如挑选室外类别，但是不希望图片中出现person
    pycocotool使用：
    https://blog.csdn.net/u013832707/article/details/94445495
    https://zhuanlan.zhihu.com/p/70878433
    """
    def __init__(self, coco_root, data_type, cats_in, cats_out):
        """
        cats_in, cats_out 分别是挑选的类别和过滤类别
        :param coco_root: 根目录， 下属应当有images 和 annotations两个目录
        :param data_type:str, val2017等
        :param cats_in: list, eg:["outdoor"]
        :param cats_out: list, eg:["person"]
        """
        self.coco_root = coco_root
        self.data_type = data_type
        self.ann_path = None
        self.coco = self._load_coco()  ###########
        self.super_cats_in = cats_in
        self.super_cats_out = cats_out
        self.cats_in_ids = None     # 全图类别id， 用于挑选图片
        self.cats_out_ids = None    # bbox目标id， 用于过滤图片
        self.img_list = []
        self.coco_super_cats = []
        self.coco_cats = []

        # 0. 加载coco类别
        self._get_coco_cats()
        # 1. 根据str获取对应图片类别id
        self._get_cats_ids()  # str --> ids
        # 2. 根据类别id获取所有的img_list(图片id)
        self._get_img_list()
        # 3. 执行过滤条件，过滤剩下的图片id
        self._filter_img_list()

    def __getitem__(self, item):
        img_id = self.img_list[item]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.coco_root, 'images', self.data_type, img_info['file_name'])
        return self.img_list[item], img_path

    def __len__(self):
        return len(self.img_list)

    def _load_coco(self):
        self.ann_path = os.path.join(self.coco_root, f'annotations/instances_{self.data_type}.json')
        coco = COCO(self.ann_path)
        return coco

    def _get_cats_ids(self):
        self.cats_in_ids = self.coco.getCatIds(supNms=self.super_cats_in)  # 选大图
        self.cats_out_ids = self.coco.getCatIds(supNms=self.super_cats_out)  # 过滤小目标

    def _get_img_list(self):
        for k, v in self.coco.catToImgs.items():
            if k in self.cats_in_ids:
                self.img_list.extend(v)
        # [self.img_list.extend(v) for k, v in self.coco.catToImgs.items() if k in self.cats_in_ids]

    def filter_func_by_id(self, img_id, cats_out_ids):
        """
        判断单张图片里的obj是否有不希望的类别
        :param img_id: 图片id
        :param cats_out_ids: 不需要类别的id
        :return:
        """
        ann_idx = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_idx)
        img_obj_idx = []
        _ = [img_obj_idx.append(a["category_id"]) for a in anns]
        is_inter = bool(set(img_obj_idx).intersection(cats_out_ids))  # 判断是否有交集
        return bool(1-is_inter)

    def _filter_img_list(self):
        filter_func = partial(self.filter_func_by_id, cats_out_ids=self.cats_out_ids)
        print(f"before filter, img length:{len(self.img_list)}")
        self.img_list = list(filter(filter_func, self.img_list))   ########
        print(f"after filter, img length:{len(self.img_list)}")

    def _get_coco_cats(self):
        for k, v in self.coco.cats.items():
            self.coco_super_cats.append(v["supercategory"])
            self.coco_cats.append(v["name"])
        self.coco_super_cats = list(set(self.coco_super_cats))

    def show_img(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.coco_root, 'images', self.data_type, img_info['file_name'])

        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im);
        plt.axis('off')

        # 获取该图像对应的anns的Id
        annIds = self.coco.getAnnIds(imgIds=img_info['id'])
        anns = self.coco.loadAnns(annIds)
        self.coco.showAnns(anns)
        plt.show()

if __name__ == '__main__':

    coco_root = "/hy-tmp/coco"
    data_type = "val2017"
    super_cats_in = ["outdoor", "indoor"]

    super_cats_out = ["person"]
    img_genertor = CocoImg(coco_root, data_type, super_cats_in, super_cats_out)

    for i in range(10):
        img_id, img_path = img_genertor[i]
        img_genertor.show_img(img_id)


