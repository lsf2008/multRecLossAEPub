#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py.py    
@Contact :   lishifeng2007@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/8 16:31   shifengLi     1.0         None
'''

# import lib
from aemodel.base import BaseModule
import torch

from torch import nn
import utils as utils
from itertools import chain

class AeBaseLoss(BaseModule):
    def __init__(self, stdInd):
        super(AeBaseLoss, self).__init__()
        self.stdInd = stdInd
    def forward(self, x, x_r):
        '''
        :param x:  list, each item is N*c*t*h*w
        :param x_r: corresponding list
        :return: list, each item is N, length=len(self.stdInd)
        '''

        if isinstance(x, list) and isinstance(x_r, list):
            x_r = list(reversed(x_r))
            enc_out = []
            dec_out = []
            for i in self.stdInd:
                enc_out.append(x[i])
                dec_out.append(x_r[i])
            ls = [(utils.cmpAeDiff(i, j)) for i, j in zip(enc_out, dec_out)]
        else:
            ls = utils.cmpAeDiff(x, x_r)
        return ls

class AeLoss(AeBaseLoss):
    def __init__(self, stdInd):
        super(AeLoss, self).__init__(stdInd)
        # self.stdInd = stdInd
    def forward(self, x, x_r):
        '''
        compute the reconstruction loss
        Parameters
        ----------
        x       orignal image (patches_per_batch, c, t, h,w )
        x_r     reconstruction image
        Returns
        -------
        '''

        ls1 = super(AeLoss, self).forward(x, x_r)
        # ls = [torch.mean(r) for r in ls1]

        # ls = torch.mean(torch.tensor(ls)).cuda().requires_grad_(True)
        ls = torch.mean(ls1[0])

        return ls
        # return torch.max(L)

class GradientLoss(BaseModule):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_r):
        # Do padding to match the  result of the original tensorflow implementation
        '''
        param:
        x: (patches_per_batch, c, t, h,w )
        x_r: reconstruction
        '''

        # ls = utils.cmpGrdDiff(x, x_r)
        ls = utils.cmpGrdDiff(x, x_r)
        return torch.mean(ls)

class TimeBaseLoss(BaseModule):
    def __init__(self, stdInd):
        super(TimeBaseLoss, self).__init__()
        self.stdInd = stdInd

    def forward(self, x, x_r):
        if isinstance(x, list) and isinstance(x_r, list):
            x_r = list(reversed(x_r))
            enc_out = []
            dec_out = []
            for i in self.stdInd:
                enc_out.append(x[i])
                dec_out.append(x_r[i])
            ls = [(utils.cmpTimGrdDiff(i, j)) for i, j in zip(enc_out, dec_out)]
            ls = ls[0]
        else:
            ls = utils.cmpTimGrdDiff(x, x_r)
        return ls
class TimeGrdLoss(TimeBaseLoss):
    def __int__(self):
        super(TimeGrdLoss, self).__int__()

    def forward(self, x, x_r):
        '''
        x, x_r   patches_per_batch*3*8*h*w
        '''
        ls = super(TimeGrdLoss, self).forward(x, x_r)

        return torch.mean(ls)


if __name__=='__main__':
    # x = torch.randn((12,3,8,40,40))
    # x_r = torch.randn((12,3,8,40,40))

    # tgl = TimeGrdLoss()
    # ael = AeLoss1()
    # print(ael(x,x_r))
    from aemodel.ae_multi_out import AeMultiOut
    input_shape = (3, 8, 32, 32)
    code_length = 64
    x = torch.randn((4, 3, 8, 32, 32))
    end = AeMultiOut(input_shape, code_length)
    x_r, z, enc_out, dec_out = end(x)
    # dec_out = reversed(dec_out)

