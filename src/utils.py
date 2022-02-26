import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

theory = pd.read_csv('../theory/theory_A_N.csv')

Rhythm = {'regular': 0, 'irregular': 1}
P = {'normal': 0, 'abnormal': 1, 'absent': 2, 'changing': 3}
RateP = {'between_100_250': 0, 'between_250_350': 1, 'between_60_100': 2, 'over_350': 3, 'under_60': 4, 'zero': 5}
P_QRS = {'after_P_always_QRS': 0, 'after_P_some_QRS_miss': 1, 'independent_P_QRS': 2, 'meaningless': 3}
PR = {'after_QRS_is_P': 0, 'changing': 1, 'meaningless': 2, 'normal': 3, 'prolonged': 4, 'shortened': 5}
Rate = {'between_100_250': 0, 'between_250_350': 1, 'between_60_100': 2, 'over_350': 3, 'under_60': 4}

theory = theory.replace({'Rhythm': Rhythm, 'P': P, 'RateP': RateP, 'P_QRS': P_QRS, 'PR': PR, 'Rate': Rate})

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx, :-1]
        sample = np.array(sample).astype(np.float64)
        sample = (sample - sample.mean()) / sample.std()   # normalize
        label = self.dataframe.iloc[idx, -1]

        return sample, label

def semantic_loss(out, labels):

    out_dict = {
        'Rhythm': out[0],
        'P': out[1],
        'RateP': out[2],
        'P_QRS': out[3],
        'PR': out[4],
        'Rate': out[5]
        }

    labels = list(labels)
    labels = [theory.loc[theory['Class'] == label].drop(['Class'], axis=1) for label in labels]

    batch_loss = []
    for b in range(out[0].shape[0]):
        loss = 0
        for i in range(labels[b].shape[0]):
            f1 = out_dict['Rhythm'][b][int(labels[b].iloc[i]['Rhythm'])]
            f2 = out_dict['P'][b][int(labels[b].iloc[i]['P'])]
            f3 = out_dict['RateP'][b][int(labels[b].iloc[i]['RateP'])]
            f4 = out_dict['P_QRS'][b][int(labels[b].iloc[i]['P_QRS'])]
            f5 = out_dict['PR'][b][int(labels[b].iloc[i]['PR'])]
            f6 = out_dict['Rate'][b][int(labels[b].iloc[i]['Rate'])]

            loss += f1 * f2 * f3 * f4 * f5 * f6
            # loss += out_dict['Rhythm'][b][labels[b].iloc[i]['Rhythm']]*out_dict['P'][labels[b].iloc[i]['P']]*out_dict['RateP'][labels[b].iloc[i]['RateP']]*out_dict['P_QRS'][labels[b].iloc[i]['P_QRS']]*out_dict['PR'][labels[b].iloc[i]['PR']]*out_dict['Rate'][labels[b].iloc[i]['Rate']]
        batch_loss.append(-torch.log(loss))
    
    total_loss = torch.mean(torch.stack(batch_loss))

    return total_loss

def metrics(out, labels, device):

    labels = list(labels)
    predictions = []

    f1 = torch.argmax(out[0], dim=-1)
    f2 = torch.argmax(out[1], dim=-1)
    f3 = torch.argmax(out[2], dim=-1)
    f4 = torch.argmax(out[3], dim=-1)
    f5 = torch.argmax(out[4], dim=-1)
    f6 = torch.argmax(out[5], dim=-1)

    for b in range(out[0].shape[0]):
        label = labels[b]

        for index, row in theory.iterrows():
            if int(f1[b]) == row['Rhythm'] and int(f2[b]) == row['P'] and int(f3[b]) == row['RateP'] and int(f4[b]) == row['P_QRS'] and int(f5[b]) == row['PR'] and int(f6[b]) == row['Rate']:
                predictions.append(row['Class'])
                break
        else:
            predictions.append('None')

    correct = 0
    for i in range(len(predictions)):
        if labels[i] == predictions[i]:
            correct += 1

    accuracy = correct / len(labels)
            
    
    return accuracy, precision, recall, f1_score