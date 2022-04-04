import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset

mapping = {
    'A': torch.tensor([[1, 5], [3, 4]]),
    'B': torch.tensor([[4, 5], [0, 1]]),
    'C': torch.tensor([[0, 3], [1, 2]]),
    'D': torch.tensor([[1, 3], [0, 4]])
}

theory = pd.read_csv('../theory/theory_synth.csv')

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
    output_1, output_2 = out

    loss = [-torch.log(output_1[i][feedback[i][0][0]] * output_2[i][feedback[i][0][1]] + output_1[i][feedback[i][1][0]] * output_2[i][feedback[i][1][1]]) for i in range(out[0].shape[0])]
    loss = torch.mean(torch.stack(loss))

    return loss

def inference(f_out, n_out, labels, label_encoder, record=False):

    labels = list(labels)
    n_predictions = []
    ns_predictions = []
    max_conjunctions = []

    # f1 = torch.argmax(out[0], dim=-1)
    # f2 = torch.argmax(out[1], dim=-1)
    # f3 = torch.argmax(out[2], dim=-1)
    # f4 = torch.argmax(out[3], dim=-1)
    # f5 = torch.argmax(out[4], dim=-1)
    # f6 = torch.argmax(out[5], dim=-1)

    probs = []
    for b in range(f_out[0].shape[0]):
        label = labels[b]
        infer = {
            'conjunct_probs': np.array([]),
            'predicted_class': np.array([]),
        }

        for index, row in theory.iterrows():
            conjunct_prob = f_out[0][b][int(row['Feature1'])] * f_out[1][b][int(row['Feature2'])]
            predicted_class = row['Class']
            infer['conjunct_probs'] = np.append(infer['conjunct_probs'], conjunct_prob.cpu().detach().numpy())
            infer['predicted_class'] = np.append(infer['predicted_class'], predicted_class)
            # if int(f1[b]) == row['Rhythm'] and int(f2[b]) == row['P'] and int(f3[b]) == row['RateP'] and int(f4[b]) == row['P_QRS'] and int(f5[b]) == row['PR'] and int(f6[b]) == row['Rate']:
            #     predictions.append(row['Class'])
            #     break
        # else:
        #     predictions.append('None')

        max_conjunctions.append(np.argmax(infer['conjunct_probs']))
        probs.append(infer['conjunct_probs'].tolist())
        ns_predictions.append(infer['predicted_class'][np.argmax(infer['conjunct_probs'])])

    ns_metrics = metrics(labels, ns_predictions)
    
    if n_out is not None:
        n_out = nn.Softmax(dim=-1)(n_out)
        n_out = torch.argmax(n_out, dim=-1).cpu().detach().tolist()
        # n_out = torch.round(n_out).flatten().to(torch.int64).cpu().detach().tolist()
        n_predictions = label_encoder.inverse_transform(n_out)
        n_metrics = metrics(labels, n_predictions)
    else:
        n_metrics = None

    if record:
        # Recording predictions
        with open('predictions.txt', 'w') as f:
            f.write('Original,NS,N\n')
            for i in range(len(labels)):
                f.write(labels[i] + ',' + ns_predictions[i] + ',' + n_predictions[i] + '\n')
        f.close()

        # Recording extracted features
        with open('extracted_features.txt', 'w') as f:
            for i in max_conjunctions:
                row = theory.iloc[[i]].to_string(header=False, index=False, index_names=False).split(' ')
                row = ','.join(row)
                f.write(row + '\n')
        f.close()

        # Recording probabilities for each conjunct
        with open('probabilities.txt', 'w') as f:
            for dp in probs:
                for conjunct in dp:
                    f.write(str(conjunct) + ',')
                f.write('\n')

    return ns_metrics, n_metrics

def metrics(labels, predictions):

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average='macro')
    rec = recall_score(labels, predictions, average='macro')

    # tp_A, fp_A, fn_A = 0, 0, 0
    # tp_N, fp_N, fn_N = 0, 0, 0

    # for l, p in zip(labels, predictions):
    #     if p == 'A':
    #         if l == 'A':
    #             tp_A += 1
    #         elif l == 'N':
    #             fp_A += 1
    #             fn_N += 1
    #     else:
    #         if l == 'N':
    #             tp_N += 1
    #         elif l == 'A':
    #             fp_N += 1
    #             fn_A += 1

    # try:
    #     prec_A = tp_A / (tp_A + fp_A)
    # except:
    #     prec_A = 0
    
    # try:
    #     rec_A = tp_A / (tp_A + fn_A)
    # except:
    #     rec_A = 0

    # try:
    #     prec_N = tp_N / (tp_N + fp_N)
    # except:
    #     prec_N = 0

    # try:
    #     rec_N = tp_N / (tp_N + fn_N)
    # except:
    #     rec_N = 0
            
    metrics = {
        'acc': acc,
        'prec': prec,
        'rec': rec,
    }

    return metrics

# def accuracy(out, labels, device):
#     label_map = {}

#     # output_1 = out[0].unsqueeze(1)
#     # output_2 = out[1].unsqueeze(1).permute(0, 2, 1)

#     # probs = output_1*output_2

#     # n = probs.shape[0]
#     # d = probs.shape[1]

#     # m = probs.view(n, -1).argmax(1)
#     # max_indices = torch.cat(((m / d).view(-1, 1), (m % d).view(-1, 1)), dim=1)
    
#     for i,key in enumerate(mapping):
#         label_map[key] = i + 1

#     y_pred1 = torch.argmax(out[0], dim=-1)
#     y_pred2 = torch.argmax(out[1], dim=-1)

#     y_pred = torch.stack((y_pred1, y_pred2), dim=-1)
#     y_pred_inverted = torch.stack((y_pred2, y_pred1), dim=-1)
#     label_pred = torch.zeros(y_pred1.shape[0], device=device)
#     num_preds = y_pred.shape[-1]
    
#     for i in mapping:
#         for val in mapping[i]:
#             val.unsqueeze_(0)
#             val = val.to(device)
#             mask = (y_pred==val) + (y_pred_inverted==val)
#             mask = (mask >= 1)
#             mask = torch.sum(mask,-1)
#             mask = mask==num_preds
#             mask = mask * label_map[i]
#             label_pred += mask
    
#     labels = [label_map[labels[i]] for i in range(len(labels))]
#     labels = torch.tensor(labels,device=device)
#     accuracy = torch.sum(labels==label_pred)
#     accuracy = accuracy / labels.shape[0]
    
#     return accuracy