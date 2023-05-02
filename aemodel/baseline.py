import torch
from sklearn.metrics import roc_auc_score
import scipy.signal as signal

import utils


def getMaxAucFrmDict(y, s, res, epoch):
    if len(y) <= 5:
        aucmax = 0
        rec_k = '0'
        if sum(y)==0:
            y[0]=1
    else:
        y[0] = 0
        aucmax = 0
        rec_k = ''
    score =[]
    lb = []
    for k, e in s.items():
        # e = utils.normalize((e))
        val_roc = roc_auc_score(y, e)
        if val_roc >= aucmax:
            aucmax = val_roc
            rec_k = k
            epoch1 = epoch
            score = e
            lb = y

    if aucmax > res['maxAuc']:
        res['maxAuc'] = aucmax
        res['coef'] = rec_k
        res['epoch'] = epoch1
        res['score'] = score
        res['label'] = lb
    res['auc'].append(aucmax)
    # print('/n', tbl)
    return res, aucmax


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


def cmb_ae_gd_scores(ae, gd, ael=[0.5, 1, 2],
                     gdl=[0.5, 1, 2]):
    s = {}
    for a in ael:
        for g in gdl:
            s[str(a) + '-' + str(g)] = ae * a + gd * g
    return s

def cmb_ae_gd_td_scores(ae, gd, td,
                        ael=[0.5, 1, 2],
                        gdl=[0.5, 1, 2],
                        tdl=[0.5, 1, 3, 5, 8]):

    s = {}
    for a in ael:
        for g in gdl:
            for t in tdl:
                s[str(a) + '-' + str(g) + '-' + str(t)] = ae * a + gd * g + td * t
    return s


def cmb_ae_td_scores(ae, td,
                        ael=[0.5, 1, 2],
                        tdl=[0.5, 1, 3, 5, 8]):

    s = {}
    for a in ael:

        for t in tdl:

            s[str(a) + '-' + str(t)] = ae * a + td * t
    return s

def cmb_ae_td_one_scores(td,  ae, one,
              tdl=[0.5, 1, 3, 5, 8],
              ael=[0.5, 1, 2],
              onel=[0.5, 1, 2, 5]):
    s = {}
    for t in tdl:
        for a in ael:
            for o in onel:
                s[str(a) + '**' + str(t) + '**' + str(o)] = t * td + a * ae + o * one
    return s
