import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
import torch

theory_original = pd.read_csv('../theory/theory_A_N.csv')

Rhythm = {'regular': 0, 'irregular': 1}
P = {'normal': 0, 'abnormal': 1, 'absent': 2, 'changing': 3}
RateP = {'between_100_250': 0, 'between_250_350': 1, 'between_60_100': 2, 'over_350': 3, 'under_60': 4, 'zero': 5}
P_QRS = {'after_P_always_QRS': 0, 'after_P_some_QRS_miss': 1, 'independent_P_QRS': 2, 'meaningless': 3}
PR = {'after_QRS_is_P': 0, 'changing': 1, 'meaningless': 2, 'normal': 3, 'prolonged': 4, 'shortened': 5}
Rate = {'between_100_250': 0, 'between_250_350': 1, 'between_60_100': 2, 'over_350': 3, 'under_60': 4}

theory = theory_original.replace({'Rhythm': Rhythm, 'P': P, 'RateP': RateP, 'P_QRS': P_QRS, 'PR': PR, 'Rate': Rate})

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx, :-1]
        sample = np.array(sample).astype(np.float64)
        label = self.dataframe.iloc[idx, -1]

        return sample, label

def deduce(features):
    f_out = [features[:, :2], features[:, 2:6], features[:, 6:12], features[:, 12:16], features[:, 16:22], features[:, 22:]]
    max_conjunctions = []
    deductions = []

    f_out = [nn.Softmax(-1)(f) for f in f_out]

    conj_probs = []

    for index, row in theory.iterrows():
        probs = f_out[0][:, row['Rhythm']] * f_out[1][:, row['P']] * f_out[2][:, row['RateP']] * f_out[3][:, row['P_QRS']] * f_out[4][:, row['PR']] * f_out[5][:, row['Rate']]
        conj_probs.append(probs)

    index_A = theory.index[theory['Class'] == 'A'].tolist()
    index_N = theory.index[theory['Class'] == 'N'].tolist()

    conj_probs = torch.stack(conj_probs, dim=1)
    max_conjunctions = torch.argmax(conj_probs, dim=1).cpu().detach().tolist()

    wt_A = conj_probs[:, index_A].sum(dim=1)
    wt_N = conj_probs[:, index_N].sum(dim=1)

    logits = torch.stack([wt_A, wt_N], dim=1)

    deductions = ['lt' if wt_A[i] > wt_N[i] else 'gt' for i in range(len(wt_A))]

    # deductions = [theory.iloc[i]['Class'] for i in max_conjunctions]

    return logits, deductions, {'conj_probs': conj_probs, 'max_conjunctions': max_conjunctions}

def semantic_loss(features):
    pass
