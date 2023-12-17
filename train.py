import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import Dataset_ETT_hour
from model_utils import get_csv_logger
from model_config import LTFModule, ModuleConfigure
from torch.utils.data import DataLoader


class LTFConfig():
    def __init__(self, pred_len) -> None:
        self.embed = 'timeF'
        self.task_name = "long_term_forecast"
        self.features = 'M'

        self.seasonal_patterns = "Monthly"
        
        self.freq = 'h'
        self.ex_chn = 4
    
        self.seq_len = 96
        self.pred_len = pred_len
        self.label_len = 0

        self.size = (self.seq_len, self.label_len, self.pred_len)

        self.batch_size = 32
        self.forecast_type = 'M'
        self.num_workers = 8
        self.timeenc = 1
        self.grad_clip_val = 1

class ETTh1_LTFConfig(LTFConfig):
    def __init__(self, pred_len) -> None:
        super().__init__(pred_len)
        self.name = "etth1"
        self.data = "ETTh1"
        self.root_path = "/home/549/zw6060/project/Training/wzyWorkspace/MSD-Mixer-main/dataset/ETT-small"
        self.data_path = "ETTh1.csv"
        self.target = ['OT']


def ltf_experiment(config, gpus):
    pl.seed_everything(2023)
    train_dataset = Dataset_ETT_hour(root_path=config.root_path, flag='train', size=config.size, features=config.forecast_type, data_path=config.data_path, target=config.target)

    val_dataset = Dataset_ETT_hour(root_path=config.root_path, flag='val', size=config.size, features=config.forecast_type, data_path=config.data_path, target=config.target)

    test_dataset = Dataset_ETT_hour(root_path=config.root_path, flag='test', size=config.size, features=config.forecast_type, data_path=config.data_path, target=config.target)

    print(f"train  {len(train_dataset)}")
    print(f"val  {len(val_dataset)}")
    print(f"test  {len(test_dataset)}")
    # epoch, batch size.  1 epoch -> all training data passed the model.  -> multiple bsz. n basz, basz: 32, num batch len(train)/32
    train_dl = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True)
    
    val_dl = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True)
    
    test_dl = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=True)


    ModelConifg = ModuleConfigure(pred_len=config.pred_len)
    model = LTFModule(ModelConifg)
    # one_batch = next(iter(train_dl))
    # loss = model.training_step(one_batch, batch_idx=0)

    monitor_metric = "val_mse"
    callbacks = []
    ckpt_callback = ModelCheckpoint(monitor=monitor_metric,
                                    save_top_k=1,
                                    mode="min")
    callbacks.append(ckpt_callback)
    es_callback = EarlyStopping(monitor=monitor_metric,
                                mode="min",
                                patience=10)
    callbacks.append(es_callback)
    logger = get_csv_logger("logs/ltf",
                            name=f"{config.name}_{config.pred_len}")
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         precision=16,
                         callbacks=callbacks,
                         logger=logger,
                         max_epochs=50,
                         gradient_clip_val=config.grad_clip_val,
                         # strategy='ddp_find_unused_parameters_true',
                         )
    trainer.fit(model, train_dl, val_dl)
    model = LTFModule.load_from_checkpoint(ckpt_callback.best_model_path)
    trainer.test(model, test_dl)


if __name__ == "__main__":
    pred_len = 96
    etth1_ltf = ETTh1_LTFConfig(pred_len)
    ltf_experiment(etth1_ltf, gpus=[0])