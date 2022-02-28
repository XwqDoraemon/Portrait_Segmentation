# -*- coding: utf-8 -*-
"""
# @file name  : seg_aug_func.py
# @author     : XueWQ
# @date       : 2021-03-14
# @brief      : albumentations中适用于图像分割的方法
"""

import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask, cmap ='gray')
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask, cmap ='gray')
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    # plt.show()
    f.savefig("/root/workspace/img_seg/aug_demo/00079_aug.png")
    # plt.imsave("/root/wor/kspace/img_seg/aug_demo/0079_aug.jpg",f)


if __name__ == '__main__':

    image = cv2.imread("/root/workspace/img_seg/aug_demo/00079.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread('/root/workspace/img_seg/aug_demo/00079_matte.png', cv2.IMREAD_GRAYSCALE)

    # step1：定义好一系列变换方法
    aug = A.Compose([
        # A.Resize(width=336, height=448),
        # A.Blur(blur_limit=7, p=0.3),  # 采用随机大小的kernel对图像进行模糊
        # A.ChannelDropout(p=1),  # 随机选择某个通道像素值设置为 0
        # A.ChannelShuffle(p=1),    # 颜色通道随机打乱 rgb --> bgr/brg/rbg/rgb/grb/gbr
        A.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.3, hue=0.2, p=1),
        # A.GaussNoise(var_limit=(100, 255), p=1)
        # A.InvertImg(p=1),
        # A.Normalize(max_pixel_value=200.0, p=1.0),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=1),
        # A.IAAAffine(scale=(0.95, 1.1), p=1),       # 缩放
        # A.IAAAffine(translate_percent=0.2, p=0.3),  # 平移
        # A.IAAAffine(rotate=(-10, 10), p=0.3),  # 旋转 ["constant","edge","symmetric","reflect","wrap"]
        # A.IAAAffine(shear=(10, 10), p=0.3),   # 错切
        # A.CoarseDropout(max_holes=5, p=1)   #  对于工业场景，适用。《Improved Regularization of Convolutional Neural Networks with Cutout》
        # A.ElasticTransform(p=1, border_mode=1),  # 2003年针对MNIST数据集提出的方法。alpha越小，sigma越大，产生的偏差越小，和原图越接近
        # A.LongestMaxSize(max_size=800, p=1),    # 依最常边保持比例的缩放
        # A.OneOf([
            # A.HorizontalFlip(p=1),
        #     A.VerticalFlip(p=1),
        #     A.Sequential([
        #         A.HorizontalFlip(p=1),
        #         A.VerticalFlip(p=1),
        #     ], p=1),
        # ], p=1),
        # A.Sequential([
            # A.HorizontalFlip(p=1),
            # A.VerticalFlip(p=1),
        # ], p=1),
    ])
    # aug = A.Compose([
    #     A.Compose([
    #         A.Blur(blur_limit=7, p=0.3),  # 采用随机大小的kernel对图像进行模糊
    #         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
    #         A.HorizontalFlip(p=0.5),  
    #         A.OneOf([
    #             A.IAAAffine(translate_percent=0.2, p=0.3),  # 平移
    #             A.IAAAffine(rotate=(-10, 10), p=0.3),  # 旋转 ["constant","edge","symmetric","reflect","wrap"]
    #         ], p=0.5)
    #     ])
        # A.Resize(width=cfg.in_size, height=cfg.in_size),
        # A.Normalize(norm_mean, norm_std),
    # ])
    # step2：给该变换出入源数据（通常在Dataset的__getitem__中使用）
    augmented = aug(image=image, mask=mask)
    # step3：获取变换后的数据 （通常在Dataset的__getitem__中使用）
    image_aug = augmented['image']
    mask_aug = augmented['mask']

    # 观察效果
    print("raw: ", image.shape, mask.shape)
    print("aug: ", image_aug.shape, mask_aug.shape)
    visualize(image_aug, mask_aug, original_image=image, original_mask=mask)



