# from aemodel.ae_multi_out import AeMultiOut
import torch

from aemodel.ae_multi_out_wghts import AeMultiOut
from config.param_seting import initial_params
from trainer.trainer import trainer_vd_module
# from aemodel.autoencoder import convAE
from dataset.video_dataloader import VideoDataLoader

from trainer.mult_ae_mot_recLoss_module import MultAeMotRecLossModule

import pytorch_lightning as pl
import prettytable as pt
import time
import utils as utils
# pl.seed_everything(999999)
flg = 'ped2'
tbl = pt.PrettyTable()
tbl.field_names = ['auc', 'cmb_coef', 'layers', 'epoch']
if flg =='ped2':
    pl.seed_everything(999999)
    stat_time = time.time()
    # ===================ae==================

    args = initial_params('config/ped2_cfg.yml')
    # 数据集
    vd = VideoDataLoader(**vars(args))
    # 模型
    model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length,
                       layers= args.wght_layers
                       )
    # module
    # --------------------only motion--------------
    mdl = MultAeMotRecLossModule(model, **vars(args))

    # 使用module训练模型
    res = trainer_vd_module(args, mdl, vd, is_test=True)
    end_time = time.time()
    tbl.add_row([res['maxAuc'], res['coef'], args.rec_layers, res['epoch']])
    print(tbl)

    print(f'all running time:{(end_time - stat_time) / 60} m')


    # ===================AE+ FINCH==================
    # stat_time = time.time()
    # args = initial_params('config/ped2_cfg.yml')
    # # 数据集
    # vd = VideoDataLoader(**vars(args))
    # # 模型
    # model = AeMultiOut(input_shape=args.input_shape,
    #                    code_length=args.code_length)
    # # module
    # mdl = MultRecLossModuleFinch(model, **vars(args))
    # # mdl = MultRecLossModule(model, **vars(args))
    #
    # # 使用module训练模型
    # res = trainer_vd_module(args, mdl, vd)
    # end_time = time.time()
    # print(f'running time:{(end_time - stat_time) / 60} m')
if flg =='ave':
    args = initial_params('config/ave_cfg.yml')
    stat_time = time.time()
    # 数据集
    vd = VideoDataLoader(**vars(args))
    # 模型
    model = AeMultiOut(input_shape=args.input_shape,
                       code_length=args.code_length,
                       layers=args.wght_layers)
    # module
    # ------------------only ae-----------------
    # mdl = MultAERecLossModule(model, **vars(args))

    # --------------------only motion--------------
    mdl = MultAeMotRecLossModule(model, **vars(args))

    # 使用module训练模型
    res = trainer_vd_module(args, mdl, vd)
    end_time = time.time()
    tbl.add_row([res['maxAuc'], res['coef'], args.rec_layers, res['epoch']])
    print(tbl)
    print(f'running time:{(end_time - stat_time) / 60} m')


if flg =='shg':

    print('shg test '.center(100, '='))
    tbl = pt.PrettyTable()
    tbl.field_names = ['auc', 'cmb_coef', 'epoch', 'dataset']
    stat_time = time.time()

    trn_pth = 'E:/dataset/shanghaitech/label/remove_bg/train/'
    trnList = utils.getImgListFrm1Flder(trn_pth, type='*.csv')

    tstPth = 'E:/dataset/shanghaitech/label/remove_bg/test/'
    aucList = []
    for trn in trnList:
        tstCSV = tstPth + 'test' + str(trn).split('_')[-1]
        args = initial_params('config/dtsht_cfg.yml')
        args.train_dataPath = str(trn)
        args.val_dataPath = tstCSV

        # 数据集
        vd = VideoDataLoader(**vars(args))

        # 模型
        model = AeMultiOut(input_shape=args.input_shape,
                           code_length=args.code_length,
                           layers=args.wght_layers)
        # trainer module
        mdl = MultAeMotRecLossModule(model, **vars(args))

        # 使用module训练模型
        res = trainer_vd_module(args, mdl, vd)

        # print(str(trn)+res['maxAuc'])
        aucList.append(res['maxAuc'])
        sn = 'data/shg/' + str(trn).split('_')[-1].split('.')[0] + '.pt'
        tbl.add_row([res['maxAuc'], res['coef'], res['epoch'], str(trn).split('_')[-1].split('.')[0]])
        torch.save(res, sn)

    end_time = time.time()

    print(tbl)
    print(f'averange auc is: {aucList}')
    print(f'running time:{(end_time - stat_time) / 60} m')
    sv_name = 'data/shg/shg_wgtLayer[034]_multLayer034.txt'
    with open(sv_name, 'w') as f:
        f.write(tbl.get_string())
