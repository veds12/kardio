import numpy as np
import torch
from torch.utils.data import Dataset

mapping = {
    'A': torch.tensor([[1, 5], [3, 4]]),
    'B': torch.tensor([[4, 5], [0, 1]]),
    'C': torch.tensor([[0, 3], [1, 2]]),
    'D': torch.tensor([[1, 3], [0, 4]])
}

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
    feedback = [mapping[labels[i]] for i in range(len(labels))]
    loss = [-torch.log(out[i][feedback[i][0][0]] * out[i][feedback[i][0][1]] + out[i][feedback[i][1][0]] * out[i][feedback[i][1][1]]) for i in range(out.shape[0])]
    loss = torch.mean(torch.stack(loss))

    return loss

def accuracy(out, labels, device):
    label_map = {}
    
    for i,key in enumerate(mapping):
        label_map[key] = i + 1

    y_pred1 = torch.argmax(out, dim=-1)
    for i in range(y_pred1.shape[0]):
        out[i][y_pred1[i]] = 0
    
    y_pred2 = torch.argmax(out, dim=-1)
    y_pred = torch.stack((y_pred1, y_pred2), dim=-1)
    y_pred_inverted = torch.stack((y_pred2, y_pred1), dim=-1)
    label_pred = torch.zeros(y_pred1.shape[0], device=device)
    num_preds = y_pred.shape[-1]
    
    for i in mapping:
        for val in mapping[i]:
            val.unsqueeze_(0)
            val = val.to(device)
            mask = (y_pred==val) + (y_pred_inverted==val)
            mask = (mask >= 1)
            mask = torch.sum(mask,-1)
            mask = mask==num_preds
            mask = mask * label_map[i]
            label_pred += mask
    
    labels = [label_map[labels[i]] for i in range(len(labels))]
    labels = torch.tensor(labels,device=device)
    accuracy = torch.sum(labels==label_pred)
    accuracy = accuracy / labels.shape[0]
    
    return accuracy