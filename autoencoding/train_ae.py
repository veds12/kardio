import argparse
import os
import warnings
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import pandas as pd
import numpy as np
import wandb

from utils import TimeSeriesDataset
from models import get_model

def train(args):

    if args.verbose:

        print('----------------------------------------------------')

        print('CONFIG\n')
        print(f'Seed: {args.seed}')
        print(f'Number of training epochs: {args.epochs}')
        print(f'Learning Rate: {args.lr}')
        print(f'Batch size: {args.batch_size}')
        print(f'Data path: {args.data_dir}')
        print(f'Run name: {args.name}')
        print(f'Checkpoint save path: {args.checkpoint}')
        print(f'Checkpoint load path: {args.load_chkpt}')
        print(f'Device: {args.device}')
        print(f'Model: {args.model_name}')
        print(f'Embedding size: {args.emb_size}')

    ######################### SETTING DEVICE AND DATA TYPE #########################

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not torch.cuda.is_available():
            warnings.warn('CUDA is not available. Using CPU instead.')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    dtype = torch.double

    ########################### SET SEEDS ###########################

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    ##################################################################################

    ######################### LOAD AND PREPROCESS DATA ###############################

    if args.verbose:
        print('----------------------------------------------------')
        print('PREPARING DATA\n')
    
    train_data_path = os.path.join(args.data_dir, 'physionet_train_data.csv')
    test_data_path = os.path.join(args.data_dir, 'physionet_test_data.csv')

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    data_N = train_data.loc[train_data['Class'] == 'N']
    data_A = train_data.loc[train_data['Class'] == 'A']
    data_O = train_data.loc[train_data['Class'] == 'O']

    # data_N = data_N[:int(0.18*data_N.shape[0])]
    # data_O = data_O[:int(0.3*data_O.shape[0])]

    # train_data = pd.concat([data_N, data_A, data_O])
    # train_data = train_data.sample(frac=1).reset_index(drop=True)

    train_min = train_data.min()[:-1].to_numpy().min()
    train_max = train_data.max()[:-1].to_numpy().max()

    columns = train_data.columns.tolist()[:-1]

    columns = train_data.columns.tolist()[:-1]
    train_data_num = train_data[columns].to_numpy()
    mean = train_data_num.mean()
    std = train_data_num.std()
    # train_data[columns] -= train_min
    # train_data[columns] /= (train_max - train_min)
    # test_data[columns] -= train_min
    # test_data[columns] /= (train_max - train_min)

    train_data[columns] -= mean
    train_data[columns] /= std
    test_data[columns] -= mean
    test_data[columns] /= std

    train_dataset = TimeSeriesDataset(train_data)
    test_dataset = TimeSeriesDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.verbose:
        print(f'Number of datapoints for class A: {data_A.shape[0]}')
        print(f'Number of datapoints for class N: {data_N.shape[0]}')
        print(f'Number of datapoints for class O: {data_O.shape[0]}')
        print(f'Number of training samples: {train_data.shape[0]}')
        print(f'Number of test samples: {test_data.shape[0]}')

    ##################################################################################

    ########################### CONFIGURE MODEL AND OPTIMIZER ########################

    model = get_model(model_name=args.model_name, embedding_dim=args.emb_size).to(device).to(dtype)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ##################################################################################

    ########################### CHECKPOINTING ########################################

    if args.load_chkpt is not None:
        model.load_state_dict(torch.load(args.load_chkpt))

    if args.checkpoint is not None:
        CHECKPOINT_PATH = os.path.join(args.checkpoint, args.name)

        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)

    ##################################################################################

    ########################### LOGGING ##############################################

    if args.logging:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            project='Kardio',
            entity='appcair-kardio',
            name=args.name,
            config=args,
        )

    ##################################################################################

    ########################### TRAINING #############################################

    if args.verbose:        
        print('----------------------------------------------------')
        print('TRAINING\n')

    for epoch in range(args.epochs):
        train_loss = []
        
        for x, labels in train_dataloader:
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            x_recon, x_emb = model(x)
            # print(x.shape)
            # print(x_recon.shape)
            loss = nn.MSELoss()(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)

        if args.logging:
            wandb.log({
                'train_loss': train_loss,
                'epoch': epoch,
            }, step=epoch)

        if args.verbose:
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.5f} | ', end='')

        if epoch % args.test_every == 0:
            test_loss = evaluate(model, test_dataloader, device, dtype)

            print(f'Test Loss: {test_loss:.5f}')
            wandb.log({'test_loss': test_loss}, step=epoch)
        else:
            print(f'Test Loss: NA')

        if args.checkpoint is not None and epoch % args.chkpt_every == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'{args.seed}.pt'))

    ##################################################################################

def evaluate(model, test_dataloader, device, dtype):
    test_loss = []
    model.eval()

    with torch.no_grad():
        for x, labels in test_dataloader:
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            x_recon, x_emb = model(x)
            print(x)
            print(x_recon)

            loss = nn.MSELoss()(x_recon, x)

            test_loss.append(loss.item())

        test_loss = sum(test_loss) / len(test_loss)
    
    model.train()
    
    return test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder Training Kardio')

    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='cnn', help='Name of model')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--test_every', type=int, default=1, help='Test every n epochs')
    parser.add_argument('--chkpt_every', type=int, default=1, help='Save checkpoint every n epochs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint directory')
    parser.add_argument('--name', type=str, default=None, help='Name of experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--logging', action='store_true', help='Log to wandb')
    parser.add_argument('--load_chkpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    args = parser.parse_args()
    train(args)


