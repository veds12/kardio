import os
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import wandb

from models import CNNModule
from utils import TimeSeriesDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.double

def train(args):

    if args.verbose:
        print('----------------------------------------------------')

        print(f'CONFIG\n')
        print(f'Seed: {args.seed}')
        print(f'Number of training epochs: {args.epochs}')
        print(f'Learning Rate: {args.lr}')
        print(f'Batch size: {args.batch_size}')
        print(f'Data path: {args.data_dir}')
        print(f'Run name: {args.name}')
        print(f'Checkpoint save path: {args.checkpoint}')
        print(f'Checkpoint load path: {args.load_chkpt}')
        print(f'Device: {device}')
        print(f'Model: {args.model_name}')

    ###################################### SET SEEDS #################################

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    ##################################################################################

    ############################### LOAD AND PREPROCESS DATA #########################

    if args.verbose:
        print('----------------------------------------------------')
        print('PREPARING DATA\n')

    train_data_path = os.path.join(args.data_dir, 'physionet_train_data.csv')
    test_data_path = os.path.join(args.data_dir, 'physionet_test_data.csv')

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_data_N = train_data.loc[train_data['Class'] == 'N']
    train_data_A = train_data.loc[train_data['Class'] == 'A']
    train_data_N = train_data_N.sample(frac=1)[:int(0.15*len(train_data_N))]

    train_data_A = train_data_A.replace(to_replace='A', value='lt')
    train_data_N = train_data_N.replace(to_replace='N', value='gt')
    train_data = pd.concat([train_data_N, train_data_A])
    
    test_data_N = test_data.loc[test_data['Class'] == 'N']
    test_data_A = test_data.loc[test_data['Class'] == 'A']
    
    test_data_A = test_data_A.replace(to_replace='A', value='lt')
    test_data_N = test_data_N.replace(to_replace='N', value='gt')
    test_data = pd.concat([test_data_N, test_data_A])

    train_min = train_data.min()[:-1].to_numpy().min()
    train_max = train_data.max()[:-1].to_numpy().max()

    columns = train_data.columns.tolist()[:-1]
    train_data[columns] -= train_min
    train_data[columns] /= (train_max - train_min)
    test_data[columns] -= train_min
    test_data[columns] /= (train_max - train_min)

    train_dataset = TimeSeriesDataset(train_data)
    test_dataset = TimeSeriesDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=False)

    if args.verbose:
        print(f'Number of datapoints for class A: {train_data_A.shape[0]}')
        print(f'Number of datapoints for class N: {train_data_N.shape[0]}')
        print(f'Number of training samples: {train_data.shape[0]}')
        print(f'Number of test samples: {test_data.shape[0]}')

    ##################################################################################

    ########################### CONFIGURE MODEL AND OPTIMIZER ########################

    model = CNNModule().to(device).to(dtype)
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
            out = model(x)

            encoded_labels = LabelEncoder().fit(['lt', 'gt']).transform(labels)
            encoded_labels = torch.tensor(encoded_labels).to(device).to(dtype).reshape(out.shape[0], 1)
            loss = nn.BCEWithLogitsLoss()(out, encoded_labels)

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
            print(f'Epoch {epoch}: Train Loss: {train_loss:.5f} | ', end='')

        if epoch % args.test_every == 0:
            test_loss, test_acc = evaluate(model, test_dataloader)

            print(f'Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.5f}')
            if args.logging:
                wandb.log({
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                })
        else:
            print(f'Test Loss: NA | Test Accuracy: NA')

        if args.checkpoint is not None and epoch % args.chkpt_every == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'{args.seed}.pt'))

    ##################################################################################

def evaluate(model, test_dataloader):
    model.eval()
    label_encoder = LabelEncoder()
    test_loss = []
    test_acc = []

    with torch.no_grad():
        for x, labels in test_dataloader:
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            out = model(x)

            encoded_labels = label_encoder.fit(['lt', 'gt']).transform(labels)
            encoded_labels = torch.tensor(encoded_labels).to(device).to(dtype).reshape(out.shape[0], 1)
            loss = nn.BCEWithLogitsLoss()(out, encoded_labels)

            out = nn.Sigmoid()(out)
            out = torch.round(out).flatten().to(torch.int64).cpu().detach().tolist()
            predictions = label_encoder.inverse_transform(out)
            acc = accuracy_score(labels, predictions)

            test_loss.append(loss.item())
            test_acc.append(acc)

    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_acc) / len(test_acc)

    model.train()

    return test_loss, test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RateP feature detector')

    parser.add_argument('--data_dir', type=str, default='../autoencoding/data/', help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='cnn', help='Name of model')
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

    args = parser.parse_args()
    train(args)

