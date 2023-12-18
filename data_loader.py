import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
# from utils.dtwCostMatrix import dtw_cost_matrix_multi, dtw_cost_matrix_pdist
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]       #96
            self.label_len = size[1]      #0
            self.pred_len = size[2]       #96
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]  #[8640, 7]
        
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]   # [96, 7]

        # dtw_cost = dtw_cost_matrix_pdist(seq_x)   # [21, 96, 96]
        # dtw_cost = 1

        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]  #[96,4]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    


if __name__ == "__main__":
    seq_len = 96
    label_len = 0
    pred_len = 120

    size =(seq_len, label_len, pred_len)
    batch_size = 32
    target = ['OT']
    forecast_type = 'MS'
    num_workers = 8
    data_root_path = "/home/549/zw6060/project/Training/wzyWorkspace/MSD-Mixer-main/dataset/ETT-small"
    data_path='ETTh1.csv'
    ex_chn = 4
    in_chn = 7
    out_chn = 1
    all_chn = in_chn + ex_chn
    pair_chn = 6

    dataset_ETT_h1 = Dataset_ETT_hour(root_path=data_root_path, flag='train', size=size, features=forecast_type, data_path=data_path, target=target, scale=True, timeenc=1, freq='h')
    print(dataset_ETT_h1.data_x.shape, type(dataset_ETT_h1.data_x))
    print(dataset_ETT_h1.data_y.shape, type(dataset_ETT_h1.data_y))

    data_loader = DataLoader(
            dataset_ETT_h1,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True)
    
    x_batch, y_batch, x_mark_batch, y_mark_batch, pairwise_batch = next(iter(data_loader))
    print(x_batch.shape)
    print(y_batch.shape)  # torch.Size([32, 96, 7])
    
    y_batch = y_batch[:,:, :out_chn] 
    print("get target:",y_batch.shape)

    print(x_mark_batch.shape)
    print(y_mark_batch.shape)
    print(pairwise_batch.shape)
    
    x_input = torch.cat((x_batch, x_mark_batch), dim=-1)
    print(x_input.shape)   # 96, 11

    x_input = x_input.float()
    pairwise_batch = pairwise_batch.float()

    from PairWiseAttnTransformer import PairWiseTransformer
    model = PairWiseTransformer(d_model=512, num_heads=8, d_ff=2048, num_layers=6, dropout=0.2, input_feasize=all_chn, dtwcost_inpsize=pair_chn,output_chn=out_chn, input_len=seq_len, output_len = pred_len)

    output = model(x_input, pairwise_batch)
    output_squeezed = output.squeeze()
    print(output_squeezed.shape)  

