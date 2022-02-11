import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import random
import pandas as pd

"""
Function generators

These are responsible for the generation of base signals for the time series 

"""
def blank(r,resolution= 0.1):
    """
    This function generates a blank signal
    """
    r = 2*r
    x = np.arange(0,r,resolution)
    y = np.zeros(x.shape)
    return x,y

def quadrant(r,resolution= 0.1):
    """
    Formation of quarter circle (top left) of radius r
    """
    r = 2*r
    x = np.arange(0,r,resolution)
    y = np.sqrt(r**2 - (x-r)**2)
    extra = np.arange(r,r+1,resolution)
    x = np.concatenate((x,extra))
    y = np.concatenate((y,np.zeros(extra.shape)))
    return x,y

def semi(r,resolution= 0.1):
    """
    Formation of semi circle (lower half) of radius r
    """
    x = np.arange(0,2*r,resolution)
    y = r-np.sqrt(r**2 - (x-r)**2)
    extra = np.arange(2*r,2*r+1,resolution)
    x = np.concatenate((x,extra))
    y = np.concatenate((y,np.zeros(extra.shape)))
    return x,y

def triangular(r,resolution= 0.1):
    """
    Formation of triangular shape of base length 2r
    """
    x1 = np.arange(0,r,resolution)
    x2 = np.arange(r,2*r,resolution)
    y = np.concatenate((x1,2*r-x2))
    extra = np.arange(2*r,2*r+1,resolution)
    x = np.concatenate((x1,x2,extra))
    y = np.concatenate((y,np.zeros(extra.shape)))
    return x,y

def bell(r,resolution= 0.1):
    """
    Formation of bell shape about r
    """
    x = np.arange(0,2*r,resolution)
    y = r*np.exp(-(x-r)**2)
    extra = np.arange(2*r,2*r+1,resolution)
    x = np.concatenate((x,extra))
    y = np.concatenate((y,np.zeros(extra.shape)))
    return x,y

def sin_triangular(r,resolution= 0.1,amplitude = 0.1,frequency = 10):
    """
    Formation of triangular shape with sin deviation of base length 2r
    """
    x,y = triangular(r,resolution)
    y+= amplitude*np.sin(frequency*(r-x))
    mask = x<2*r
    y*= mask
    return x,y

def square(r,resolution= 0.1):
    """
    Formation of square graph of width r
    """
    x = np.arange(0,2*r,resolution)
    y = x<r
    y = y*2-1
    extra = np.arange(2*r,2*r+1,resolution)
    x = np.concatenate((x,extra))
    y = np.concatenate((y,np.zeros(extra.shape)))
    return x,y

"""
Time-Signal generators
These functions are used for combining the generated signals for the final time series
"""
def merge(x1,y1,x2,y2):
    """
    This function merges two signals into one
    """
    x = np.concatenate((x1,x2+x1[-1]))
    y = np.concatenate((y1,y2))
    return x,y

def duplicate(x,y,n):
    """
    This function duplicates the signal n times
    """
    xt = x
    yt = y
    for i in range(n):
        xt,yt = merge(xt,yt,x,y)
    return xt,yt

func_dict = {0:blank,1:semi,2:triangular,3:bell,4:square,5:quadrant}

def add_noise(datax,std = 0.05):
    """
    Add gaussian noise to the data
    """
    return np.random.normal(datax,std,datax.shape)

def generate_timeseries(r,n,seed,resolution= 0.1):
    """
    The function generates a time series repeating n times
    with 2 randomly selected curves alternating with each other.
    """
    x1,y1 = func_dict[seed[0]](r,resolution)
    x2,y2 = func_dict[seed[1]](r,resolution)
    x,y = merge(x1,y1,x2,y2)
    x,y = duplicate(x,y,n)
    y = add_noise(y)
    return np.expand_dims(x,0),np.expand_dims(y,0),seed


def generate_dataset(r,n,resolution=0.1):
    """
    Compiles all the time series into a dataset along with 
    their constituent feature keys
    """
    seed = list(combinations(range(0,6),2))
    random.shuffle(seed)
    datax = generate_timeseries(r,n,seed[0],resolution)[0][:,:10000]
    datay = generate_timeseries(r,n,seed[0],resolution)[1][:,:10000] 
    for i in seed[1:8]:
        datax = np.concatenate((datax,generate_timeseries(r,n,i,resolution)[0][:,:10000]))
        datay = np.concatenate((datay,generate_timeseries(r,n,i,resolution)[1][:,:10000]))
    return datax*10,datay,seed[:8]

def plot_dataset(dataset):
    """
    Plotting function to visualize the generated dataset
    """
    plt.figure(figsize=(40,20))
    for i in range(1,9):
        plt.subplot(4,2,i)
        plt.plot(dataset[0][i-1][:300,0],dataset[0][i-1][:300,1])
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title(f'Signal with {dataset[1][i-1][0]} and {dataset[1][i-1][1]}')
    plt.show()

def convert_to_csvs(dataset):
    """
    Export generated dataset to csv files
    """
    for i in range(len(dataset[0])):
        DF = pd.DataFrame(dataset[0][i])
        DF.to_csv(f'time_series_{dataset[1][i][0]}_and_{dataset[1][i][1]}.csv',
                    header=['time_step','signal'],index=False)


dataset = generate_dataset(2,125)
seeds = dataset[2]
dataset = [np.expand_dims(dataset[i],-1) for i in range(2)]

#merging the time step and signal into a single numpy array
dataset = [np.concatenate([dataset[0],dataset[1]],axis=-1)]
dataset.append(seeds)
