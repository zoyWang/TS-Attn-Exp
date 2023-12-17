import torch
from torch.optim.lr_scheduler import *
import pytorch_lightning as pl
from model_utils import get_loss_fn
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from MyModel import MyNet
import torch.nn.functional as F


class ModuleConfigure():
    def __init__(self, pred_len) -> None:
        self.optim = 'adam'
        self.lr = 1e-3
        self.lr_factor = 0.5
        self.weight_decay = 0.1
        self.patience = 1
        self.loss_fn = 'mse'

        self.in_chn = 7
        self.ex_chn = 4
        self.pair_chn = 6
        self.out_chn = 7

        self.seq_len = 96
        self.pred_len = pred_len
        self.label_len = 0

        self.d_model = 512
        self.num_heads= 8
        self.d_ff = self.d_model*4
        # self.layers = 6
        self.dropout = 0.2

class LTFModule(pl.LightningModule):

    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.out_chn = config.out_chn
        self.optim = config.optim
        self.lr = config.lr
        self.lr_factor = config.lr_factor
        self.weight_decay = config.weight_decay
        self.patience = config.patience
        self.patch_len = 16
        self.stride = 8
    

        self.model = MyNet(seq_len=config.seq_len, pred_len=config.pred_len, in_chn=config.in_chn, ex_chn=config.ex_chn, out_chn=config.out_chn, d_model= config.d_model, num_heads = config.num_heads, d_ff=config.d_ff, dropout=config.dropout)

        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.loss_fn = get_loss_fn(config.loss_fn)

    def training_step(self, batch, batch_idx):
        # x, y, x_mark, _ = batch
        x, y, x_mark, _ = batch
        x = x.float()
        y = y.float()
        x_mark = x_mark.float() 
    
        y = y[:, :, :self.out_chn]

        y_pred = self.model(x)


        pred_loss = self.loss_fn(y_pred, y)

        # residual_loss = residual_loss_fn(res, self.lambda_mse, self.lambda_acf,
        #                                self.acf_cutoff)
        loss = pred_loss
        # self.log('train_loss', pred_loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y, x_mark, _ = batch
        x = x.float()
        y = y.float()
        y = y[:, :, :self.out_chn]

        x_mark = x_mark.float()

        y_pred = self.model(x)

        self.val_mse(y_pred, y)
        self.val_mae(y_pred, y)
        self.log("val_mse", self.val_mse)
        self.log("val_mae", self.val_mae)
        self.log("lr", self.optimizers().param_groups[0]['lr'])
        return

    def test_step(self, batch, batch_idx):
        x, y, x_mark, _ = batch
        x = x.float()
        y = y.float()
        y = y[:, :, :self.out_chn]
        x_mark = x_mark.float()
    
        y_pred = self.model(x)
        
        self.test_mse(y_pred, y)
        self.test_mae(y_pred, y)
        self.log("test_mse", self.test_mse)
        self.log("test_mae", self.test_mae)

        return

    def configure_optimizers(self):
        if self.optim == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            raise ValueError
        scheduler_config = {}
        # scheduler_config["scheduler"] = ExponentialLR(optimizer,0.5)
        scheduler_config["scheduler"] = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.lr_factor,
            patience=self.patience,
            verbose=False,
            min_lr=1e-8)
        scheduler_config["monitor"] = "val_mse"
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }





