'''
Description: 
Autor: Shifeng Li
Date: 2022-10-12 11:36:46
LastEditTime: 2022-10-15 07:44:32
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage.morphology import binary_dilation
import cv2
import torch.nn  as nn
import os
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py.py    
@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/12 11:36   shifengLi     1.0         None
'''

# import lib
import yaml
import torch
import pytorch_lightning.callbacks as plc
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
from pathlib import Path
import cv2
from dataset.video_dataloader import VideoDataLoader
from torchvision.transforms._transforms_video import NormalizeVideo
import argparse
def initial_params(train_cfg='config/ave_train_cfg.yaml', dt_cfg='config/dtped2_cfg.yml'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', type=str, default=train_cfg)
    parser.add_argument('--dt_cfg', type=str, default=dt_cfg)

    # --------------debug--------------
    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--precision', type=int, default=16)

    # 配置优化策略
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'cosine'], default='step')

    args = parser.parse_args()

    train_cfg = load_config(args.train_cfg)
    dt_cfg = load_config(args.dt_cfg)
    train_cfg['input_shape'] = dt_cfg['input_shape']
    train_cfg['batch_size'] = dt_cfg['batch_size']
    train_cfg['max_epochs'] = args.max_epochs

    args.callbacks = load_callbacks(args)
    return args, train_cfg, dt_cfg

def shwMaxAucFrmDic(dict):
    '''
    show the maximum auc from the dictionary dict
    :param dict: dictionary whose key is combination of loss funcion coefficients and corresponding
    value is also a dictionary containing maxAuc, epoch, coef, auc, and label
    :return:
    '''
    for key, value in dict.items():
        print(key, end=':')
        print(value['maxAuc'])
def cmp_normalize_coef(dt):
    '''
    1d tensor
    return: min and max of the tensor
    '''
    if len(dt) != 0:
        return torch.max(dt), torch.min(dt)


def normalize(samples):
    # type: (np.ndarray, float, float) -> np.ndarray
    """
    Normalize scores as in Eq. 10

    :param samples: the scores to be normalized.
    :param min: the minimum of the desired scores.
    :param max: the maximum of the desired scores.
    :return: the normalized scores
    """
    if len(samples) != 0:
        maxs, mins = cmp_normalize_coef(samples)
        # if maxs-mins!=0:
        return (samples - mins) / (maxs - mins + 1e-7)


def cmpAeDiff(x, x_r):
    '''
    :param x:   original data (patches_per_batch, c, t, h, w)
    :param x_r: reconstruction data (patches_per_batch, c, t, h, w)
    :return:  vector with size of patches_per_batch
    '''
    b, c, t, h, w = x.shape
    x = x.reshape((b, t, c, h, w))
    x_r = x_r.reshape((b, t, c, h, w))
    r = torch.mean(torch.pow((x - x_r), 2), dim=list(range(x.dim() - 1, 0, -1)))
    # r = nn.MSELoss(reduction='mean')(x, x_r)
    return r


def load_config(filePth):
    with open(filePth, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
        return config


def cmpGrd(x):
    '''
    compute the x and y gradient, (patches_per_batch, c, t, h, w)
    return b*h*w
    '''
    b, c, t, h, w = x.shape
    x = x.reshape((b * c * t, h, w))
    xi_1 = torch.zeros(x.shape, dtype=torch.float).to(x.device)
    xi_1[:, 0:-1, :] = x[:, 1:, :]
    x_d = torch.abs(x - xi_1)
    x_d = x_d.reshape((b, c, t, h, w))

    yj_1 = torch.zeros(x.shape, dtype=torch.float).to(x.device)
    yj_1[:, :, 0:-1] = x[:, :, 1:]
    y_d = torch.abs(x - yj_1)
    y_d = y_d.reshape((b, c, t, h, w))
    # return torch.mean(x_d, dim=0), torch.mean(y_d, dim=0)
    return x_d, y_d


def cmpGrdDiffL2(x, x_r):
    '''
    :param x: (patches_per_batch, c, t, h, w)
    :param x_r: (patches_per_batch, c, t, h, w)
    :return:
    '''
    x_gx, x_gy = cmpGrd(x)
    xr_gx, xr_gy = cmpGrd(x_r)
    ls = torch.pow((x_gx - xr_gx), 2) + torch.pow((x_gy - xr_gy), 2)
    ls = torch.mean(ls, dim=list(range(x.dim() - 1, 0, -1)))
    return ls


def cmpGrdDiff(x, x_r):
    '''

    :param x: (patches_per_batch, c, t, h, w)
    :param x_r: (patches_per_batch, c, t, h, w)
    :return:
    '''

    # (b,c,t,h,w)
    x_gx, x_gy = cmpGrd(x)
    xr_gx, xr_gy = cmpGrd(x_r)
    ls = torch.abs(x_gx - xr_gx) + torch.abs(x_gy - xr_gy)
    ls = torch.mean(ls, dim=list(range(x.dim() - 1, 0, -1)))
    return ls


def cmpTimGrd(x):
    '''
    :param x:   tensor with(b,c,t,h,w)
    :return: tensor (b, c, t-1, h, w)
    '''
    # b, c, t, h, w = x.shape
    # x = x.reshape((b, t, c, h, w))
    xi = x[:, :, 1:, :, :]
    xj = x[:, :, 0:-1, :, :]

    return torch.abs(xi - xj)


def initTable(cols=['epoch', 'aeCoef', 'gdCoef', 'tdgCoef', 'oneCoef', 'AUC']):
    table = PrettyTable()
    table.field_names =cols
    table.float_format = '0.3'
    return table


def inertTblScores(epoch, s, y, tbl):
    '''
    insert into the table with dictionary score s
    :param epoch:   current epoch
    :param s:  dict of scores
    :param y:  ground truth
    :return:
    return the optimal auc and the corresponding coefficients
    '''
    if len(y) <= 5:
        aucmax = 0
        rec_k = '0'
    else:
        y[0] = 1
        aucmax = 0
        rec_k = ''
        for k, e in s.items():
            val_roc = roc_auc_score(y, e)
            if val_roc > aucmax:
                aucmax = val_roc
                rec_k = k
            cols = k.split('-')
            cols.insert(0, str(epoch))
            tbl.add_row(cols + [val_roc])
        print('/n', tbl)
    return aucmax, rec_k


def cmbScores(td, gd, ae, one,
              tdl=[0.5, 1, 3, 5, 8],
              gdl=[0.5, 1, 2],
              ael=[0.5, 1, 2],
              onel=[0.5, 1, 2, 5]):
    s = {}
    for t in tdl:
        for g in gdl:
            for a in ael:
                for o in onel:
                    s[str(t) + '-' + str(g) + '-' + str(a) + '-' + str(o)] = t * td + g * gd + a * ae + o * one
    return s


def cmpTimGrdDiffL2(x, x_r):
    '''

     :param x: tensor with(b,c,t,h,w)
     :param x_r: tensor with(b,c,t,h,w)
     :return: scalar (b,)
     '''
    x_t = cmpTimGrd(x)
    xr_t = cmpTimGrd(x_r)
    # b, c, t, h, w
    ls = torch.pow((x_t - xr_t), 2)
    ls = torch.mean(ls, dim=list(range(x.dim() - 1, 0, -1)))
    return ls


def cmpTimGrdDiff(x, x_r):
    '''
    :param x: tensor with(b,c,t,h,w)
    :param x_r: tensor with(b,c,t,h,w)
    :return: scalar (b,)
    '''
    x_t = cmpTimGrd(x)
    xr_t = cmpTimGrd(x_r)
    # b, c, t, h, w
    ls = torch.abs(x_t - xr_t)
    ls = torch.mean(ls, dim=list(range(x.dim() - 1, 0, -1)))
    return ls


def log10(t):
    """
    Calculates the base-10 tensorboard_log of each element in t.
    @param t: The tensor from which to calculate the base-10 tensorboard_log.
    @return: A tensor with the base-10 tensorboard_log of each element in t.
    """
    numerator = torch.log(t)
    denominator = torch.log(torch.FloatTensor([10.])).cuda()
    return numerator / denominator


def psnr_error(gen_frames, gt_frames):
    """
    Computes the Peak Signal to Noise Ratio error between the generated images and the ground
    truth images.
    @param gen_frames: A tensor of shape [batch_size, 3, height, width]. The frames generated by the
                       generator model.
    @param gt_frames: A tensor of shape [batch_size, 3, height, width]. The ground-truth frames for
                      each frame in gen_frames.
    @return: A scalar tensor. The mean Peak Signal to Noise Ratio error over each frame in the
             batch.
    """
    shape = list(gen_frames.shape)  # [b, c, t, h,w ]
    num_pixels = (shape[2] * shape[4] * shape[5])
    gt_frames = (gt_frames + 1.0) / 2.0  # if the generate ouuput is sigmoid output, modify here.
    gen_frames = (gen_frames + 1.0) / 2.0
    square_diff = (gt_frames - gen_frames) ** 2
    # maxf=torch.max(gen_frames)
    batch_errors = 10 * log10(1. / ((1. / num_pixels) * torch.sum(square_diff, [2, 4, 5])))
    return 1 / (torch.mean(batch_errors, [2, 1]) + 1e-7)


def load_callbacks(args):
    '''define the training callbacks

    Parameters
    ----------
    args : argument
        parameters defined by user

    Returns
    -------
    callbacks
        callbacks for earlystopping+modelcheckpoint+lr_scheduler
    '''
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_roc',
        mode='max',
        patience=20,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_roc',
        filename='best-{epoch:02d}-{val_roc:.3f}',
        save_top_k=1,
        mode='max',

        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'
        ))
    return callbacks


def getImgsFrmMulFlder(pth, imtype='*.tif'):
    frames = []
    fld = Path(pth)
    for flder in fld.iterdir():
        if flder.is_dir():
            imgNameList = getImgListFrm1Flder(str(flder), type=imtype)
            frames.extend(imgNameList)
    return frames


def getImgListFrm1Flder(pth, type='*.tif'):
    """ get the image name list from pth
    Args:
        pth ([string]): [path of frame path]
        type (str, optional): [description]. Defaults to '*.tif'.
    Returns:
        [list]: [image name list]
    """
    imgNamePth = Path(pth)
    imgL = imgNamePth.glob(type)
    return list(imgL)


def cmpBg(pth='E:/dataset/UCSD/UCSDped2/Train/', type='*.tif', svPth='dataset/bg/ped2Bg.pt'):
    mog = cv2.createBackgroundSubtractorMOG2()
    imglst = getImgsFrmMulFlder(pth, type)
    # bg = cv2.cvtColor(cv2.imread(str(imglst[0])))
    bg1 = np.zeros_like((cv2.imread(str(imglst[0]))), dtype=np.float)
    for imgN in imglst:
        # bg += cv2.cvtColor(cv2.imread(str(imglst[i])), cv2.COLOR_BGR2RGB)/255.0
        print(imgN)
        # h,w,3
        img = (cv2.imread(str(imgN)))
        # img = (img - [0.45, 0.45, 0.45])/([0.225, 0.225, 0.225])
        # img = nor(img)
        bg1 += img
        mog.apply(img)
    bg = mog.getBackgroundImage()
    bg1 = bg1 / len(imglst)
    if svPth is not None:
        torch.save(bg, svPth)
        torch.save(bg1, 'dataset/bg/a.pt')
    return bg, bg1


def loadDatasetBg(pth):
    bg = torch.tensor(torch.load(pth))
    # if len(bg.shape)==3:
    #     bg = bg.permute([2, 0, 1])
    #     bg = torch.unsqueeze(bg, 1)
    # else:

    return bg


def cmpMeanStdFrmLoder(train_dl):
    means = torch.zeros((3))
    stds = torch.zeros((3))

    for i, f in enumerate(train_dl):
        # b1, b2, c, t, h, w
        x = f['video']
        x = x.reshape((-1, *x.shape[-4:]))
        for d in range(3):
            means[d] += x[:, d, :, :, :].mean()
            stds[d] += x[:, d, :, :, :].std()
    means = means / (i + 1)
    stds = stds / (i + 1)
    # print(means, stds)
    return means, stds
def cmpMeanStd(cfgPth='config/ped2_cfg.yml'):
    dt_cfg = load_config(cfgPth)
    vd = VideoDataLoader(**dt_cfg)
    train_dl = vd.train_dataloader()
    # c, t, h, w = dt_cfg['crop_size']
    means, stds = cmpMeanStdFrmLoder(train_dl)
    print(means, stds)
    # torch.save(mean)

def cmpShanghaiTechBg(pth):
    '''
    compute the background on shanghai technique  university dataset
    :param pth: label path of '.csv'
    :return:
    '''
    from pathlib2 import Path
    fl = Path(pth).iterdir()
    for l in (fl):
        print(l)
        fls = pd.read_csv(l)
        imFlds = [str(x).split(' ')[0] for x in fls.iloc[:, 0]]
        # imgList = getImgsFrmMulFlder(imgPth)
        aveImg = torch.zeros((480, 856, 3))
        cnt = 0
        for imFld in imFlds:
            imNames = getImgListFrm1Flder(imFld, type='*.jpg')
            for imName in imNames:
                img = (cv2.imread(str(imName)))
                aveImg += img
                cnt += 1
        aveImg /= cnt

        svPth = str(l).split('\\')[-1].split('.')[0] + '_bg.pt'
        svPth = 'dataset/bg/' + svPth
        torch.save(aveImg, svPth)

def getMaxAuc(y, preds, res, epoch):
    '''
    :param y:  array of gt
    :param preds: array of prediction
    :param res: dict ['maxAuc']
    :return: res
    '''
    if sum(y) == 0:
        y[0] = 1
    elif sum(y) == len(y):
        y[0] = 0

    val_roc = roc_auc_score(y, preds)
    if res['maxAuc'] < val_roc:
        res['maxAuc'] = val_roc
        res['epoch'] = epoch
    return val_roc

def cmpShanghaiTechMeanStd(labelPth):
    args, train_cfg, dt_cfg = initial_params(train_cfg='config/ave_train_cfg.yaml', dt_cfg='config/dtsht_cfg.yml')
    from pathlib2 import Path

    # training labels
    trn_flist = Path(labelPth).iterdir()
    meanStdDic = {}
    for f in trn_flist:  # f = image folder
        print(f)
        dt_cfg['train_dataPath'] = str(f)
        bgPth = str(f).split('\\')[-1].split('.')[0] + '_bg.pt'
        dt_cfg['bgPth'] = 'dataset/bg/' + bgPth

        # dt_cfg['mean'] =
        # dt_cfg['std'] =
        vd = VideoDataLoader(**dt_cfg)
        train_dl = vd.train_dataloader()

        means, stds = cmpMeanStdFrmLoder(train_dl)

        meanStd = torch.cat((means, stds), dim=0)
        key = str(f).split('\\')[-1].split('.')[0].split('_')[-1]
        meanStdDic[key] = meanStd

    df = pd.DataFrame(meanStdDic)
    # df.to_excel('config/sht_mean_std.xls')

def filterMask(mask, minAreaThr=280,
            maxAreaThr = 20000,
            whRadio = 7):
    '''
    filter the binary mask
    :param mask:  h*w ndarray
    :param minAreaThr: minimum area
    :param maxAreaThr: maximum area
    :param whRadio:
    :return:
    '''
    kenel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(np.uint8(mask), kenel)
    # processing mask
    num_labels, labels, stats, centeroids = cv2.connectedComponentsWithStats(mask)

    for i in range(1, num_labels):
        # area, one row is x,y, w,h, area
        x, y, w, h, area = stats[i]
        nw, nh = min(w, h), max(w, h)
        if area < minAreaThr or area > maxAreaThr or nh // nw > whRadio:
            msk = labels == i
            mask[msk] = 0
    mask = cv2.dilate(np.uint8(mask), kenel)

    return mask
def removeShtechBg(pth=r'E:/dataset/shanghaitech/training/frames', preStr='train', thr=128):
    import matplotlib.pyplot as plt
    flds=Path(pth).iterdir()
    for fld in flds:
        # fld is a video
        bgName = preStr+'_'+str(fld).split('\\')[-1].split('_')[0]+'_bg.pt'
        bgName ='dataset/bg/'+bgName
        # load the background
        bg = torch.load(bgName)

        resFld = '/'.join(pth.split('/')[0:-1])+'/rebg_frames/'+str(fld).split('\\')[-1]
        if not Path(resFld).exists():
            Path(resFld).mkdir()

        imgList = getImgListFrm1Flder(str(fld), type='*.jpg')
        for imN in imgList:
            im = cv2.cvtColor(cv2.imread(str(imN)), cv2.COLOR_BGR2RGB)
            mask = np.sum(np.abs(im - bg.numpy()), axis =-1)>thr
            mask = binary_dilation(mask, iterations=5)
            # mask = filterMask(mask)

            mask = mask.reshape((*mask.shape, 1))
            im = mask*im
            svName = resFld+'/'+str(imN).split('\\')[-1]
            cv2.imwrite(svName, im)
def cmpBgCv2(pth='E:/dataset/UCSD/UCSDped2/Train/',
             type='*.tif',
             svPth='dataset/bg/',
             shwImg=False):
    mog = cv2.bgsegm.createBackgroundSubtractorGSOC()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    imglst = getImgsFrmMulFlder(pth, type)
    for imgN in imglst:
        print(imgN)

        fldPth = svPth + '\\' + str(imgN).split('\\')[-2]
        if not os.path.exists(fldPth):
            os.mkdir(fldPth)
        imgName = fldPth + '\\' + str(imgN).split('\\')[-1]
        # h,w,3
        img = (cv2.imread(str(imgN)))
        fg_mask = mog.apply(img)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        fg_mask = fg_mask > 0
        fg_mask = fg_mask.reshape((*fg_mask.shape, 1))
        im = img * fg_mask
        if shwImg:
            plt.imshow(im)
            plt.show()
            plt.pause(0.6)
            plt.close()
        cv2.imwrite(imgName, im)

if __name__ == '__main__':
    # x = torch.randn((5, 3, 8, 32, 32)).cuda()
    # x_r = torch.randn((5, 3, 8, 32, 32)).cuda()

    # print(psnr_error(gen_frames, gt_frames))

    # ------------time grade ---------
    # print(cmpTimGrd(gen_frames).shape)
    # ---------------gradient -------------
    # print(cmpGrdDiff(x, x_r))
    # ---------------time gradient---------
    # print(cmpTimGrdDiff(x, x_r))

    # ----------combine socres--------------
    # ae = torch.tensor([1, 2, 3])
    # gd = torch.tensor([0, 1, 2])
    # tdg = torch.tensor([3, 2, 1])
    # one = torch.tensor([1, 1, 1])
    # s = cmbScores(tdg, gd, ae, one)
    # print(s)
    # -------------table show--------------
    # tbl = initTable()
    # # s={'1':[1,2,3], '2':[0,2,1]}
    # y = [0, 1, 0]
    # inertTblScores(s, y, tbl)
    # -----------background image----------------
    # bpth='E:/dataset/Avenue/Train/'
    # svpth = 'dataset/bg/avenueBg.pt'
    # bg, bg1 = cmpBg(bpth, '*.jpg', svpth)
    # print(np.max(bg), np.min(bg))
    # bg = loadDatasetBg('dataset/bg/avenueBg.pt')
    # print(bg.shape)
    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # plt.imshow(bg)
    # plt.subplot(122)
    # plt.imshow(bg1)
    # plt.show()
    # ---------shanghaitech---------------
    # pth = 'E:/dataset/shanghaitech/label/train/'
    # cmpShanghaiTechBg(pth)


    # --------------mean std-----------------
    cfgPth = 'config/ped2_cfg.yml'
    # cfgPth = 'config/ped2_cfg.yml'
    cmpMeanStd(cfgPth)


    # -----------------removeShtechBg--------------
    # pth = 'E:/dataset/shanghaitech/testing/frames'
    # removeShtechBg(pth, preStr='train', thr=128)
    # im = cv2.cvtColor(cv2.imread('dataset/bg/1009.jpg'), cv2.COLOR_BGR2RGB)
    # torch.save(im, 'dataset/bg/train_01_bg1.pt')

    # ---------------compute background---------------
    # pth = r'E:\dataset\Avenue\Test'
    # type = '*.jpg'
    # svPth = r'E:\dataset\Avenue\rebg\Test'
    # shwImg = False
    # cmpBgCv2(pth, type, svPth, shwImg)