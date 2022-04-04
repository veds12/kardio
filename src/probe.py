import torch
import torch.nn
from models import CNNModule
from utils import TimeSeriesDataset, inference, semantic_loss
from torch.utils.data import DataLoader
import os
from models import CNNModule
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(46)
random.seed(46)
np.random.seed(46)
torch.backends.cudnn.benchmark = False

model_path = './checkpoint/cnn_neural_branch_final/46.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.double
model = CNNModule(neural_branch=True).to(device).to(dtype)

model.load_state_dict(torch.load(model_path))

data_path = '../data/physionet_A_N_rescaled.csv'
data = pd.read_csv(data_path)

data_N = data.loc[data['CLASS'] == 'N']
data_A = data.loc[data['CLASS'] == 'A']

data_N = data_N.sample(frac=0.2)

data = pd.concat([data_N, data_A])

data = data.sample(frac=1)
data = data.dropna()


train_data = data[:int(0.9*data.shape[0])]
test_data = data[int(0.9*data.shape[0]):]

train_dataset = TimeSeriesDataset(train_data)
test_dataset = TimeSeriesDataset(test_data)

test_dataloader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=True)

model.eval()
label_encoder = LabelEncoder()

for x, labels in test_dataloader:
    x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
    features_out, neural_out = model(x)
    
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    encoded_labels = label_encoder.transform(labels)
    encoded_labels = torch.tensor(encoded_labels, dtype=neural_out.dtype, device=neural_out.device).reshape(neural_out.shape[0], 1)

    ns_metrics, n_metrics = inference(features_out, neural_out, labels, label_encoder, record=True)