import os
import argparse
import warnings
import time
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
from utils import  semantic_loss, TimeSeriesDataset, metrics

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
            entity='appcair-kardio',
            name=args.name,
            config=args,
        )

    ################# DATA PREPROCESSING AND LOADING ######################

    if args.verbose:
        print('----------------------------------------------------')
        print('PREPARING DATA\n')
    
    rescale_factor = 0.1

    data = pd.read_csv(args.data)

    data_N = data.loc[data['CLASS'] == 'N']
    data_A = data.loc[data['CLASS'] == 'A']

    data_N = data_N.sample(frac=0.2)

    data = pd.concat([data_N, data_A])

    data = data.sample(frac=1)
    data = data.dropna()


    train_data = data[:int(0.9*data.shape[0])]
    test_data = data[int(0.9*data.shape[0]):]

    train_dataset = TimeSeriesDataset(train_data)
    test_dataset = TimeSeriesDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=True)


    if args.verbose:
        print(f'Number of datapoints for class A: {data_A.shape[0]}')
        print(f'Number of datapoints for class N: {data_N.shape[0]}')
        print(f'Number of training samples: {train_data.shape[0]}')
        print(f'Number of test samples: {test_data.shape[0]}')

    ########################################################################

    ######################## TRAINING ######################################

    if args.verbose:        
        print('----------------------------------------------------')
        print('TRAINING\n')

    for epoch in range(args.epochs):

        train_loss = []
        l = 0
        for x, labels in train_dataloader:
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            start_1 = time.time()
            out = model(x)
            end_1 = time.time()
            loss = semantic_loss(out, labels)
            end_2 = time.time()

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

            l += 1

            if args.logging:
                wandb.log({
                    'step_loss': loss.item()
                })
            
            print(f'Forward pass: {end_1 - start_1:.4f}s | Loss computation: {end_2 - end_1:.4f}s')
        
        train_loss = sum(train_loss) / len(train_loss)
        if args.logging:
            wandb.log({
                'train_loss': train_loss,
                'epoch': epoch,
            })

        if args.verbose:
            print(f'Epoch: {epoch} | Train Loss: {train_loss:.3f} | ', end='')

        if epoch % args.test_every == 0:
            test_loss, avg_acc, avg_prec, avg_rec, prec_A, rec_A, prec_N, rec_N  = evaluate(model, test_dataloader, device, dtype)
            # test_loss = evaluate(model, test_dataloader, device, dtype)
            
            if args.logging:
                wandb.log({
                    'test_loss': test_loss,
                    'test_accuracy': avg_acc,
                    'avg_test_precision': avg_prec,
                    'avg_test_recall': avg_rec,
                    'test_precision_A': prec_A,
                    'test_recall_A': rec_A,
                    'test_precision_N': prec_N,
                    'test_recall_N': rec_N,
                })
            
            if args.verbose:
                print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {avg_acc:.3f} | Test Precision: {avg_prec:.3f} | Test Recall: {avg_rec:.3f}')
                # print(f'Test Loss: {test_loss:.3f}')

        else:
            if args.verbose:
                # print('Test Loss: NA | Test Accuracy: NA')
                print('Test Loss: NA | Test Accuracy: NA | Test Precision: NA | Test Recall: NA')

        if args.checkpoint is not None:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'{args.seed}.pt'))

    ########################################################################

def evaluate(model, test_dataloader, device, dtype):
    model.eval()

    with torch.no_grad():
        test_loss = 0
        acc_list = []
        prec_list = []
        rec_list = []
        prec_A_list = []
        rec_A_list = []
        prec_N_list = []
        rec_N_list = []

        for x, labels in test_dataloader:
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            out = model(x)
            loss = semantic_loss(out, labels)
            test_loss += loss.detach()
            acc, prec, rec, prec_A, prec_N, rec_A, rec_N = metrics(out, labels)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            prec_A_list.append(prec_A)
            rec_A_list.append(rec_A)
            prec_N_list.append(prec_N)
            rec_N_list.append(rec_N)

        avg_acc = sum(acc_list) / len(acc_list)
        avg_prec = sum(prec_list) / len(prec_list)
        avg_rec = sum(rec_list) / len(rec_list)
        prec_A = sum(prec_A_list) / len(prec_A_list)
        rec_A = sum(rec_A_list) / len(rec_A_list)
        prec_N = sum(prec_N_list) / len(prec_N_list)
        rec_N = sum(rec_N_list) / len(rec_N_list)

    model.train()
    return test_loss, avg_acc, avg_prec, avg_rec, prec_A, rec_A, prec_N, rec_N

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Module')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='cnn', help='Type of neural module. Choose from: lstm, cnn, mlp')
    parser.add_argument('--data', type=str, default='../data/physionet_A_N_rescaled.csv', help='path of data')
    parser.add_argument('--name', type=str, default=None, help='name of run')
    parser.add_argument('--checkpoint', type=str, default=None, help='path for storing model checkpoints')
    parser.add_argument('--load_chkpt', type=str, default=None, help='path for loading model checkpoints')
    parser.add_argument('--test_every', type=int, default=3, help='number of epochs between testing')
    parser.add_argument('--logging', type=bool, default=False, help='logging to wandb')
    parser.add_argument('--verbose', type=bool, default=False, help='print verbose')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    args = parser.parse_args()
    train(args)