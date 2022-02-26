import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import get_model
from utils import accuracy, semantic_loss, TimeSeriesDataset

def train(args):

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    dtype = torch.double

    ######################## PRINT CONFIG ##############################

    if args.verbose:

        print('----------------------------------------------------')

        print('CONFIG\n')
        print(f'Seed: {args.seed}')
        print(f'Model: {args.model}')
        print(f'Number of training epochs: {args.epochs}')
        print(f'Learning Rate: {args.lr}')
        print(f'Batch size: {args.batch_size}')
        print(f'Stride: {args.stride}')
        print(f'Data path: {args.data}')
        print(f'Run name: {args.name}')
        print(f'Checkpoint save path: {args.checkpoint}')
        print(f'Checkpoint load path: {args.load_chkpt}')
        print(f'Device: {device}')

    ######################################################################

    ################## CONFIGURE MODEL AND OPTIMIZER #####################

    model = get_model(args.model)().to(device).to(dtype)
    model.train()

    opt = optim.Adam(model.parameters(), lr=args.lr)

    ######################################################################

    ######################## CHECKPOINTING ###############################

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt))
    
    if args.checkpoint is not None:
        CHECKPOINT_PATH = os.path.join(args.checkpoint, args.name)

        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)

    #######################################################################

    ######################## LOGGING ######################################

    if args.logging:    
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            project='Kardio',
            name=args.name,
            config=args,
        )

    ################# DATA PREPROCESSING AND LOADING ######################

    if args.verbose:
        print('----------------------------------------------------')
        print('PREPARING DATA\n')
    files = os.listdir(args.data)
    classes = ['A', 'B', 'C', 'D']
    data = pd.DataFrame()

    for i, file in enumerate(files):
        data_f = pd.read_csv(os.path.join(args.data, file))[['value']]
        j = 0

        while j + 128 < data_f.shape[0]:
            row = data_f[j:j+128].values
            row = np.reshape(row, (1, 128))
            row = np.append(row, [[classes[int(i/2)]]], axis=1)
            data = data.append(pd.DataFrame(row))

            j += args.stride

    data = data.sample(frac=1).reset_index(drop=True)

    train_data = data[:int(0.9*data.shape[0])]
    test_data = data[int(0.9*data.shape[0]):]

    pd.read_csv

    train_dataset = TimeSeriesDataset(train_data)
    test_dataset = TimeSeriesDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=True)

    if args.verbose:
        print(f'Number of training samples: {train_data.shape[0]}')
        print(f'Number of test samples: {test_data.shape[0]}')

    ########################################################################

    ######################## TRAINING ######################################

    if args.verbose:        
        print('----------------------------------------------------')
        print('TRAINING\n')

    for epoch in range(args.epochs):

        train_loss = []

        for x, labels in train_dataloader:
            
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            out = model(x)
            loss = semantic_loss(out, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

            if args.logging:
                wandb.log({
                    'step_loss': loss.item()
                })
        
        train_loss = sum(train_loss) / len(train_loss)
        if args.logging:
            wandb.log({
                'train_loss': train_loss,
                'epoch': epoch,
            })

        if args.verbose:
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | ', end='')

        if epoch % args.test_every == 0:
            test_loss, test_acc = evaluate(model, test_dataloader, device, dtype)
            
            if args.logging:
                wandb.log({
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                })
            
            if args.verbose:
                print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.3f}')

        else:
            if args.verbose:
                print('Test Loss: NA | Test Accuracy: NA')

        if args.checkpoint is not None:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'{args.seed}.pt'))

    ########################################################################

def evaluate(model, test_dataloader, device, dtype):
    model.eval()

    with torch.no_grad():
        test_loss = 0
        acc = []
        for x, labels in test_dataloader:
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            out = model(x)
            loss = semantic_loss(out, labels)
            test_loss += loss.detach()
            acc.append(accuracy(out,labels,device).detach())

        test_acc = sum(acc) / len(acc)

    model.train()
    return test_loss, test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Module')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='lstm', help='Type of neural module. Choose from: lstm, cnn, mlp')
    parser.add_argument('--data', type=str, default='./data/raw/', help='path of data')
    parser.add_argument('--name', type=str, default=None, help='name of run')
    parser.add_argument('--checkpoint', type=str, default=None, help='path for storing model checkpoints')
    parser.add_argument('--load_chkpt', type=str, default=None, help='path for loading model checkpoints')
    parser.add_argument('--test_every', type=int, default=1, help='number of epochs between testing')
    parser.add_argument('--logging', type=bool, default=False, help='logging to wandb')
    parser.add_argument('--verbose', type=bool, default=False, help='print verbose')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--stride', type=int, default=10, help='stride for sliding window')

    args = parser.parse_args()
    train(args)