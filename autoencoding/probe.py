import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import RecurrentAutoencoder, ConvolutionalAutoencoder
from utils import TimeSeriesDataset

device = torch.device('cuda')
dtype = torch.double

# Load model
model = ConvolutionalAutoencoder(seq_len=270, n_features=1, embedding_dim=64).to(device).to(dtype)
model.load_state_dict(torch.load('./checkpoints/cnn-ae-batch/42.pt'))
model.eval()

# Load data
train_data = pd.read_csv('./data/physionet_train_data.csv')
test_data = pd.read_csv('./data/physionet_test_data.csv')

columns = train_data.columns.tolist()[:-1]

columns = train_data.columns.tolist()[:-1]
train_data_num = train_data[columns].to_numpy()
mean = train_data_num.mean()
std = train_data_num.std()

# train_data[columns] -= train_min
# train_data[columns] /= (train_max - train_min)
# test_data[columns] -= train_min
# test_data[columns] /= (train_max - train_min)

# train_data[columns] -= mean
# train_data[columns] /= std
test_data[columns] -= mean
test_data[columns] /= std

test_dataset = TimeSeriesDataset(test_data)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for x, label in test_dataloader:
    x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
    x_recon, x_emb = model(x)

    x_recon = x_recon.squeeze().detach().cpu().numpy()
    x = x.squeeze().detach().cpu().numpy()

    plt.figure()
    plt.plot(x)
    plt.plot(x_recon)
    plt.savefig('./reconstruction_cnn_batch.png')
    plt.close()

    break