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


