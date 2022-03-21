import pandas as pd
from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt

legend = {}
window_length = 2700
label_np = {}
inverse_legend = {}

def load_data(data_path):
    data = loadmat(data_path)
    return data

def sliding_window(data,length=2700):
    max_length = (data.shape[1]//length)*length
    data = np.reshape(data[:,:max_length],(-1,length))
    return data

def add_label(data,label):
    label = np.array(legend[label])
    label = label.repeat(data.shape[0])
    label = np.expand_dims(label,axis=1)
    data = np.concatenate((data,label),axis=1)
    return data

def convert_to_csv(data,title):
    data = pd.DataFrame(data)
    data.to_csv('physionet_data/'+title+'.csv',index=False)

def merge_data(labels):
    assert len(labels)>1, "Need more than one label"
    data = label_np[labels[0]]
    for i in labels[1:]:
        data = np.concatenate((data,label_np[i]),axis=0)
    return data

reference_df = pd.read_csv('training2017/Reference.csv', header=None)
reference_df.columns = ['id', 'label']

for index,label in enumerate(reference_df.label.unique()):
    legend[label] = index
    inverse_legend[index] = label
    label_np[label] = np.zeros((1,window_length),dtype=np.float32)

print(legend)
for i in range(len(reference_df)):
    label = reference_df.iloc[i]['label']
    reading_id = reference_df.iloc[i]['id']
    data_path = 'training2017/' + reading_id + '.mat'
    data = load_data(data_path)
    data = data['val']
    data = sliding_window(data,window_length)
    label_np[label] = np.concatenate((label_np[label],data),axis=0)
    break

for i in label_np:
    label_np[i] = label_np[i][1:,:]
    label_np[i] = add_label(label_np[i],i)
    convert_to_csv(label_np[i],i)

train_data = merge_data(['N','A'])
convert_to_csv(train_data,'train (N and A only)')