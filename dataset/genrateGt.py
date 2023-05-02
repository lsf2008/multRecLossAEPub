#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   genrateGt.py.py    
@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/17 11:46   shifengLi     1.0         None
'''

# import lib
import numpy as np
from pathlib2 import Path
from scipy.io import loadmat
import cv2
import os
from scipy.io import loadmat
def genTrainLb(pth,lbPth='E:/dataset/UCSD/UCSDped2/label/train.csv', ftpye='*.jpg'):
    # get the folder
    flderList = list(Path(pth).iterdir())
    with open(lbPth, mode='w+') as f:

        for fld in flderList:
            flen=len(list(Path(fld).glob(ftpye)))
            lb = '0 '*flen
            pth = str(fld)+' '+lb
            pth = pth[:-1]
            f.write(pth)
            f.write('\n')
            print(len(pth))
        f.close()

def judgeLb (imgPth):
    im = cv2.imread(str(imgPth))
    s = np.sum(im)
    if s > 0:
        return 1
    else:
        return 0

def judgeFlderLb(flderPth, ftype='*.bmp'):
    '''
    one folder path for image ground truth
    '''
    flb = ''
    flist = (list(Path(flderPth).glob(ftype)))
    for fn in flist:
        flg = judgeLb(fn)
        if flg==1:
            flb+='1 '
        else:
            flb+='0 '
    return flb

def genTestLb(pth, lbPth='E:/dataset/UCSD/UCSDped2/label/test.csv'):
    flderList = list(Path(pth).iterdir())
    with open(lbPth, mode='w+') as f:
        for fld in flderList:
            if os.path.isdir(fld):

                if len(str(fld).split('\\')[-1])==10: # images
                    flb = judgeFlderLb(str(fld))
                    pth = str(fld)[:-3] + ' ' + flb
                    pth = pth[:-1]
                    f.write(pth)
                    f.write('\n')
                # print(len(pth))
        f.close()

def gen1FlderAvegt(lbPth):
    '''
    lbPth: mat file path
    '''
    m = loadmat(lbPth)
    # m['volLabel'][0].shape
    # str(m['volLabel'][0])
    s = ''
    for i in range(m['volLabel'].shape[1]):
        # print(m['volLabel'][0][i])
        if np.sum(m['volLabel'][0][i]) == 0:
            # print(m)
            s += '0 '
        else:
            s += '1 '
    return s
def genAveTstLb(imgPth, lbPth='E:/dataset/Avenue/ground_truth_demo/testing_label_mask/',
                lbSvPth='E:/dataset/AVENUE/Avenue/label/test.csv'):
    flderList = list(Path(imgPth).iterdir())
    with open(lbSvPth, 'w+') as f:
        for fld in flderList:
            if os.path.isdir(fld):
                # flen = len(list(Path(fld).glob(imgType)))
                lbName = str(int(str(fld)[-2:]))+'_label.mat'
                lbName = lbPth+'/'+lbName
                lb = gen1FlderAvegt(lbName)
                lb = lb[:-1]
                cnt = str(fld) +' '+ lb
                f.write(cnt)
                f.write('\n')
        f.close()

def genShTrainLb(pth,lbPth='E:/dataset/shanghaitech/label/', ftpye='*.jpg'):
    flderLst = ['0'+str(x) for x in list(range(1, 14)) if x< 10]
    flderLst2 = [ str(x)  for x in list(range(10, 14))]
    flderLst.extend(flderLst2)

    # filder list
    flderList = list(Path(pth).iterdir())
    for flg in flderLst:
        proc_list = []
        for fld in flderList:
            if str(fld).split('\\')[-1][:2] == flg:
                proc_list.append(fld)

        # obtain one group, such 01-*
        svLbPth = lbPth+ str(flg)+'.csv'
        with open(svLbPth, mode='w+') as f:
            for fld in proc_list:
                flen = len(list(Path(fld).glob(ftpye)))
                lb = '0 ' * flen
                pth = str(fld) + ' ' + lb
                pth = pth[:-1]
                f.write(pth)
                f.write('\n')
        f.close()


def genShTstLb(imgPth, lbPth='E:\\dataset\\shanghaitech\\testing\\test_frame_mask',
                lbSvPth='E:/dataset/shanghaitech/label/test'):
    flderLst = ['0' + str(x) for x in list(range(1, 12)) if x < 10]
    flderLst2 = [str(x) for x in list(range(10, 12))]
    flderLst.extend(flderLst2)

    # filder list
    flderList = list(Path(imgPth).iterdir())
    for flg in flderLst:
        proc_list = []
        for fld in flderList:
            if str(fld).split('\\')[-1][:2] == flg:
                proc_list.append(fld)

        # obtain one group, such 01-*
        svLbPth = lbSvPth + str(flg) + '.csv'
        with open(svLbPth, mode='w+') as f:
            for fld in proc_list:
                lbpth = lbPth+'\\'+str(fld).split('\\')[-1]+'.npy'
                lb = np.load(lbpth)

                pth = str(fld) + ' ' + ' '.join([str(x) for x in list(lb)])
                # pth = pth[:-1]
                f.write(pth)
                f.write('\n')
        f.close()

if __name__=='__main__':
    # ----------------ped2 -----------------
    # pth = 'E:/dataset/UCSD/UCSDped2/Train/'
    # lbPth = 'E:/dataset/UCSD/UCSDped2/label/train.csv'
    # genTrainLb(pth, lbPth, ftpye='*.tif')

    pth = 'E:/dataset/UCSD/UCSDped2/Test/'
    genTestLb(pth)

    # ============================avenue=======================
    # avePth = 'E:/dataset/Avenue/Train'
    # genTrainLb(avePth, 'E:/dataset/Avenue/label/train.csv', '*.jpg')

    # avePth = 'E:/dataset/Avenue/Test'
    # genTrainLb(avePth, 'E:/dataset/AVENUE/Avenue/labbel/test.csv', '*.jpg')
    # genAveTstLb(avePth)

    # =======================shanghaitec===================
    # shtPth = 'E:/dataset/shanghaitech/training/frames/'
    # genShTrainLb(shtPth, 'E:/dataset/shanghaitech/label/train_', '*.jpg')
    # genTrainLb(shtPth, 'E:/dataset/shanghaitech/label/train.csv')
    # shtPth = 'E:/dataset/shanghaitech/testing/frames/'
    # genShTstLb(shtPth)