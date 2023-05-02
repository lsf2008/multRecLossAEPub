import yaml
import torch
import argparse
import pytorch_lightning.callbacks as plc
def load_config(filePth):
    with open(filePth, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f.read())
        return config

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

# dictionary to argparse.Namespace
def dic2Namespce(parser , dic):
    for k, v in dic.items():
        parser.add_argument('--' + k, default=v)
    return parser


def initial_params(cfgPth='ped2_cfg.yml'):
    parser = argparse.ArgumentParser()

    parser.add_argument('--fast_dev_run', type=bool, default=False)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--precision', type=int, default=16)

    # 配置优化策略
    # parser.add_argument('--lr_scheduler', type=str, choices=['step', 'cosine'], default='step')

    train_cfg = load_config(cfgPth)
    parser = dic2Namespce(parser, train_cfg)

    args = parser.parse_args()

    args.callbacks = load_callbacks(args)

    return args

if __name__=='__main__':
    # train_cfg = load_config('ped2_cfg.yml')
    initial_params()