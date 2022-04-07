import os
import argparse
import warnings
import random
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import get_model
from utils import inference, semantic_loss, TimeSeriesDataset

def log_to_wandb(test_loss, ns_metrics, n_metrics):
    
    if n_metrics is not None:
        wandb.log({
                'test_loss': test_loss,
                'ns_test_accuracy': ns_metrics['acc'],
                'ns_test_precision': ns_metrics['prec'],
                'ns_test_recall': ns_metrics['rec'],
                'n_test_accuracy': n_metrics['acc'],
                'n_test_precision': n_metrics['prec'],
                'n_test_recall': n_metrics['rec'],
                })
    else:
        wandb.log({
                'test_loss': test_loss,
                'ns_test_accuracy': ns_metrics['acc'],
                'ns_test_precision': ns_metrics['prec'],
                'ns_test_recall': ns_metrics['rec'],
                })

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

    ######################## SET SEED ####################################

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    ################## CONFIGURE MODEL AND OPTIMIZER #####################

    model = get_model(args.model)(
        neural_branch=args.neural_branch
    ).to(device).to(dtype)
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
            features_out, neural_out = model(x)
            loss = semantic_loss(features_out, labels)

            if args.neural_branch:
                encoded_labels = LabelEncoder().fit(['A', 'B', 'C', 'D']).transform(labels)
                encoded_labels = torch.tensor(encoded_labels, dtype=torch.long, device=neural_out.device)
                loss += nn.CrossEntropyLoss()(neural_out, encoded_labels)
                
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
            test_loss, ns_metrics, n_metrics = evaluate(model, test_dataloader, device, dtype, record=True)
            
            if args.logging:
                log_to_wandb(test_loss, ns_metrics, n_metrics)
            
            if args.verbose:
                ns_acc = round(ns_metrics['acc'], 3)

                if n_metrics is not None:
                    n_acc = round(n_metrics['acc'], 3)
                else:
                    n_acc = 'NA'

                print(f'Test Loss: {test_loss:.3f} | NS Test Accuracy: {ns_acc} | N Test Accuracy: {n_acc}')
        else:
            if args.verbose:
                # print('Test Loss: NA | Test Accuracy: NA')
                print('Test Loss: NA | NS Test Accuracy: NA | N Test Accuracy: NA')

        if args.checkpoint is not None:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'{args.seed}.pt'))

    ########################################################################

def evaluate(model, test_dataloader, device, dtype, record=False):
    model.eval()
    label_encoder = LabelEncoder()

    with torch.no_grad():
        test_loss = 0
    
        for x, labels in test_dataloader:
            x = x.permute(1, 0).unsqueeze(-1).to(device).to(dtype)
            features_out, neural_out = model(x)
            loss = semantic_loss(features_out, labels)

            if args.neural_branch:
                label_encoder.fit(['A', 'B', 'C', 'D'])
                encoded_labels = label_encoder.transform(labels)
                encoded_labels = torch.tensor(encoded_labels, dtype=torch.long, device=neural_out.device)
                loss += nn.CrossEntropyLoss()(neural_out, encoded_labels)        # Loss of the neural branch            
            
            test_loss += loss.detach()
            ns_metrics, n_metrics = inference(features_out, neural_out, labels, label_encoder, record=record)

    model.train()
    return test_loss, ns_metrics, n_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Module')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='cnn', help='Type of neural module. Choose from: lstm, cnn, mlp')
    parser.add_argument('--neural_branch', type=bool, default=True, help='Whether to use the neural branch or not')
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