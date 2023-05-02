import pytorch_lightning
import torch

import utils
from aemodel.loss import AeLoss, TimeGrdLoss
from aemodel.abnEval import AeScore, TimeGrdScore
from sklearn.metrics import roc_auc_score
import torch.optim.lr_scheduler as lrs
import itertools
from trainer import module_utils

class MultAeMotRecLossModule(pytorch_lightning.LightningModule):
    def __init__(self, inputModel, **kwargs):
        super(MultAeMotRecLossModule, self).__init__()
        self.save_hyperparameters(ignore='inputModel')
        self.model = inputModel
        # if not next(self.model.parameters()).is_cuda:
        #     self.model=self.model.cuda()
        # loss function

        layers = self.hparams.rec_layers
        self.aeLoss = AeLoss(layers)
        self.motLoss = TimeGrdLoss(layers)

        self.aeScore = AeScore(layers, batch_size=self.hparams.batch_size)
        self.motScore = TimeGrdScore(self.hparams.batch_size)
        # test results
        self.res={'maxAuc':0, 'coef':0}


    def forward(self, x):

        x_r, z, enc_out, dec_out = self.model(x)
        x_r=self.model(x)
        z = z.reshape(z.shape[0], -1)
        return x_r
    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalide lr_scheduler type!')
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch['video']
        # x = module_utils.filterCrops(x)

        x = x.reshape((-1, *x.shape[2:]))
        # x_r= self(x)
        x_r, z, enc_out, dec_out = self.model(x)

        aeLss = self.aeLoss(enc_out, dec_out)
        motLss = self.motLoss(enc_out, dec_out)
        # print(f'------------x_r:{x_r.requires_grad},x:{x.requires_grad}--------------')
        logDic ={'aeLss': aeLss,
                 'motLss': motLss}
        self.log_dict(logDic, prog_bar=True)

        allLss = aeLss*self.hparams.aeLsAlpha+ \
                 motLss*self.hparams.motLsAlpha
        return allLss

    def validation_step(self, batch, batch_idx):
        # aeScore, y = self.tst_val_step(batch)
        return self.tst_val_step(batch)

    def validation_epoch_end(self, outputs):
        self.tst_val_step_end(outputs, logStr='val_roc')
    def test_step(self, batch, batch_idx):
        return self.tst_val_step(batch)
    def test_epoch_end(self, outputs):
        return self.tst_val_step_end(outputs, logStr='tst_roc')

    # ====================== my functions=========================
    def tst_val_step(self, batch):
        x = batch['video']
        y = batch['label']
        # x = module_utils.filterCrops(x) # n, c, t, h, w
        x = x.reshape((-1, *x.shape[2:]))
        x_r, z, enc_out, dec_out = self(x)
        # x_r = self(x)

        # calculte anomaly score
        aeScore = self.aeScore(x, x_r)
        motScore = self.motScore(x, x_r)

        if aeScore.requires_grad == False:
            aeScore = aeScore.requires_grad_()

        return aeScore, motScore, y

    def tst_val_step_end(self, outputs, logStr='val_roc'):
        # obtain all scores and corresponding y
        scores, y = module_utils.obtAllScoresFrmOutputs(outputs)
        # self.res['epoch'] = self.current_epoch

        # compute auc, scores: dictory
        module_utils.cmpCmbAUCWght(scores, y_true=y,
                                   weight=self.hparams.cmbScoreWght,
                                   res=self.res,
                                   epoch=self.current_epoch)

        print()
        print(f'optimal coef:{self.res["coef"]}'.center(100,'-'))
        self.log(logStr, self.res['maxAuc'], prog_bar=True)
