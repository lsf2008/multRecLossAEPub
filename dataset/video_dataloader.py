#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   video_dataset.py    
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/7/12 10:42   gxrao      1.0         None
'''
import cv2
import pytorch_lightning
# import torch
import pytorchvideo.data as vd
import torch.utils.data as td
import matplotlib.pyplot as plt
import os

import numpy as np
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
    RandAugment,

)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomRotation,
    RandomHorizontalFlip,
    ColorJitter,
    Grayscale,
    ToTensor,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from dataset.my_video_dataset import labeled_video_dataset
import torch
import math

from dataset.transform import FilterCrops, ToCrops, RemoveBackground, ShortScaleImgs, MyColorJitter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def updtRawShape(raw_shape, shortSide_size):
    c, t, h, w = raw_shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * shortSide_size))
        new_h = new_h // 8 * 8
        new_w = shortSide_size
    else:
        new_h = shortSide_size
        new_w = int(math.floor((float(w) / h) * shortSide_size))
        new_w = new_w // 8 * 8
    raw_shape = (c, t, new_h, new_w)
    return raw_shape


def loadDatasetBg(pth, h, w):
    '''
    :param pth: background data path
    :param h:   resized image height
    :param w:   resized image width
    :return:    resized background h*w*c
    '''
    bg = (torch.load(pth))
    # bg = bg.permute([2, 0, 1])
    # bg = torch.unsqueeze(bg, 1)

    bg = cv2.resize(np.uint8(bg), (w, h))
    # bg = bg.permute([2, 0, 1])
    # # c, 1, h, w
    # bg = torch.unsqueeze(bg, 1)
    return bg


class VideoDataLoader(pytorch_lightning.LightningDataModule):
    def __init__(self, **kwargs):
        super(VideoDataLoader, self).__init__()

        self.save_hyperparameters(ignore='inputModel')
        self.hparams.raw_shape = updtRawShape(self.hparams.raw_shape, self.hparams.shortSide_size)
        self.hparams.side_shape = self.hparams.raw_shape[2:]

        self.clip_duration = self.hparams.num_frames / self.hparams.frames_per_second
        self.stride = self.hparams.stride / self.hparams.frames_per_second
        if self.hparams.bgPth is not None:
            h, w = self.hparams.raw_shape[2:]
            bg = loadDatasetBg(self.hparams.bgPth, h, w)
            # [c, t, h,w]
            self.bg = bg

        else:
            self.bg = None

    def train_dataloader(self):
        train_transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    # ToGrays(),
                    # 关于transform的介绍参考https://blog.csdn.net/irving512/article/details/118637392
                    #                 # 在指定的[min_size, max_size] 范围内随机取值，作为短边的长度，然后按比例缩放
                    #                 # 去掉，加上没有什么区别
                    # UniformTemporalSubsample(self.hparams.num_frames),
                    # Lambda(lambda x: x / 255.0),
                    ShortScaleImgs(size=self.hparams.side_shape),
                    # RemoveBackground(128, self.bg),

                    Lambda(lambda x: x / 255.0),
                    # Normalize(self.hparams.mean, self.hparams.std),
                    NormalizeVideo(self.hparams.mean, self.hparams.std, True),
                    ToCrops(self.hparams.raw_shape, self.hparams.input_shape),
                    # FilterCrops(0.01),
                ]
            ),
        )
        sampler = self.clipSampler(self.hparams.train_dt_sampler)

        train_dataset = self.make_dataset(self.hparams.train_dataPath, sampler, train_transform)

        return td.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_works,
            drop_last=True
        )

    def clipSampler(self, sampler_type):
        if sampler_type == 'uniform':
            clip_sampler = vd.make_clip_sampler('uniform', self.clip_duration, self.stride)
        if sampler_type == 'random':
            clip_sampler = vd.make_clip_sampler('random', self.clip_duration)
        return clip_sampler

    def make_dataset(self, dtpth, clip_sampler, transform):
        '''
        build the dataset with the path dtpth using the special clip_sampler and transform
        :param dtpth:
        :param clip_sampler:
        :param transform:
        :return:
        '''
        val_dataset = labeled_video_dataset(
            data_path=dtpth,
            clip_sampler=clip_sampler,
            decode_audio=False,
            transform=transform,
            decoder=None
        )
        return val_dataset

    def val_testLoader(self, dtpth):
        '''for validation or test
        Parameters
        ----------
        dtpth : str, optional
            which csv file should be val.csv or test.csv, by default 'val.csv'
        Returns
        -------
        dataloader
            test or validation dataloader
        '''
        val_transform = Compose(
            [ApplyTransformToKey(
                key='video',
                transform=Compose(
                    [
                        # ToGrays(),
                        # UniformTemporalSubsample(self.hparams.num_frames),

                        ShortScaleImgs(size=self.hparams.side_shape),
                        # CenterCropVideo(input_shape=(self.hparams.input_shape, self.hparams.input_shape)),
                        # NormalizeVideo(self.hparams.mean, self.hparams.std),
                        # RemoveBackground(128, self.bg),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(self.hparams.mean, self.hparams.std, True),
                        ToCrops(self.hparams.raw_shape, self.hparams.input_shape),
                        # FilterCrops(0.01),
                    ]
                )
            )]
        )

        clip_sampler = self.clipSampler(self.hparams.val_dt_sampler)
        val_dataset = self.make_dataset(dtpth, clip_sampler, val_transform)

        return td.DataLoader(val_dataset,
                             batch_size=self.hparams.batch_size,
                             num_workers=self.hparams.num_works,
                             drop_last=True)

    def val_dataloader(self):
        return self.val_testLoader(dtpth=self.hparams.val_dataPath)

    def test_dataloader(self):
        return self.val_testLoader(dtpth=self.hparams.tst_dataPath)


if __name__ == '__main__':
    import utils
    import numpy as np

    cfgPth = r'..\config\dtped2_cfg.yml'
    dt_cfg = utils.load_config(cfgPth)
    dt_cfg['bgPth'] = r'E:\program\python\paper\3DAE\dataset\bg\ped2Bg.pt'
    # print(dt_cfg)
    vd1 = VideoDataLoader(**dt_cfg)
    trainDl = vd1.train_dataloader()

    for i, x in enumerate(trainDl):
        print(f'----i:{i}, x shape:{x["video"].shape}, y shape:{x["label"].shape}')
        # b, c, t, h, w
        im = x["video"]
        im = im[0, 0, :, 0, :, :]
        # im = im[0,:, 0, :, :]
        im = im.permute(1, 2, 0)
        plt.imshow((im * torch.tensor(dt_cfg['std']) + torch.tensor(dt_cfg['mean'])))
        plt.show()

    # print(x['video'].permute(0, 2, 3, 4, 1)[0, 0, :].shape)
    # import matplotlib.pyplot as plt
    # import os
    # # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # im = x['video'].permute(0, 2, 3, 4, 1)[0, 0, :]
    # plt.imshow(im, cmap='')
    # plt.show()
    # ----------remove bg----------------
    # x = torch.rand((3, 16, 240, 360))
    # bg = loadDatasetBg('bg/ped2Bg.pt', 240, 360)
    # # print(torch.max(torch.tensor(bg)))
    # r=RemoveBackground(128, torch.tensor(bg))
    # print(r(x).shape)
    # plt.imshow(bg)
    # plt.show()

    # -------------to gray---------------
    # gry = ToGrays()
    # x = torch.rand((5, 3, 8, 256, 256))
    # y = gry(x)
    # print(y.shape)
