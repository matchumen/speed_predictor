import sys, os
sys.path.insert(1, '../../')
sys.path.insert(1, '../../../')
import random

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import optuna
import pickle
from tqdm import tqdm
import utils as utils
from sklearn.preprocessing import MinMaxScaler


def define_model(params):
    layers = [
        ('fc1', nn.Linear(params['INPUT_SIZE'], 42)), ('relu1', getattr(nn, params['ACTIVATION_NAME'])()),
        ('fc2', nn.Linear(42, 21)), (f'relu2', getattr(nn, params['ACTIVATION_NAME'])()),
        ('fc3', nn.Linear(21, 10)), (f'relu3', getattr(nn, params['ACTIVATION_NAME'])()),
        ('fc4', nn.Linear(10, 1)), ('sigmoid', nn.Sigmoid())
    ]

    model_rna = OrderedDict(layers)
    model = nn.Sequential(model_rna)

    return model


def objective(trial):
    # define model and move to device
    ACTIVATION_NAME = trial.suggest_categorical('ACTIVATION_NAME', ['ELU', 'ReLU', 'LeakyReLU', 'LogSigmoid', 'Sigmoid'])
    model = define_model({'INPUT_SIZE': train_data[0][0].shape[0], 'ACTIVATION_NAME': ACTIVATION_NAME})
    model = model.double().to(device)

    # oproti 0 jsem snížil, jde to pak jak raketa dolů a všecho co jde pomaleji to prunuje
    LR = trial.suggest_float('LEARNING_RATE', 1.0e-7, 1.0e-4, log=True)

    n_of_epochs = 50

    OPTIMIZER_NAME = trial.suggest_categorical('OPTIMIZER', ['RMSprop']) #'SGD', 'AdamW', 'Rprop', 'Adam', 

    criterion = nn.L1Loss().double()
    optimizer = getattr(optim, OPTIMIZER_NAME)(model.parameters(), lr=LR)

    train_history = [1]
    test_history = [1]
    best_loss = 1
    best_model = model

    for epoch in tqdm(range(n_of_epochs)):
        model.train()
        train_loss = 0
        val_loss = 0

        for _, data in enumerate(trainloader, 0):
            sample, label = data
            sample = sample.double().to(device)
            label = label.double().to(device)
            loss = 0

            pred = model(sample)

            loss = criterion(pred.squeeze(), label.squeeze())

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        trl = train_loss / len(trainloader)
        train_history.append(trl)

        model.eval()
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                sample, label = data
                sample = sample.double().to(device)
                label = label.double().to(device)
                
                loss = 0

                pred = model(sample)

                loss = criterion(pred.squeeze(), label.squeeze())

                val_loss += loss.item()

        vll = val_loss / len(testloader)
        test_history.append(vll)

        if vll < best_loss:
            best_model = model
            best_loss = vll

        trial.report(best_loss, epoch)

        if trial.should_prune():
            # save history
            with open(f'history/history_{best_loss:.6f}.pickle', 'wb') as f:
                pickle.dump((train_history, test_history), f)

            raise optuna.exceptions.TrialPruned()

    # save model
    torch.save(best_model.state_dict(), f'models/model_{best_loss:.6f}.pth')

    # save history
    with open(f'history/history_{best_loss:.6f}.pickle', 'wb') as f:
        pickle.dump((train_history, test_history), f)
    
    return best_loss


if __name__ == '__main__':
    # set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # set target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare data
    BATCH_SIZE = 1024

    # load drivea
    dataset_dir = '../../data/train_dataset_small'
    drive_fns = os.listdir(dataset_dir)
    random.shuffle(drive_fns)

    # make train val split
    train_size = int(len(drive_fns) * 0.8)
    
    train_drives = drive_fns[:train_size]
    train_drives = [os.path.join(dataset_dir, d) for d in train_drives]
    
    val_drives = drive_fns[train_size:]
    val_drives = [os.path.join(dataset_dir, d) for d in val_drives]

    train_dfs = utils.load_preprocess_drives(train_drives, drop_null_cols=True, blur_loc_features=True, add_stop_feature=False)
    val_dfs = utils.load_preprocess_drives(val_drives, drop_null_cols=True, blur_loc_features=True, add_stop_feature=False)

    #fit scaler to all train dfs
    scaler = MinMaxScaler()
    scaler.fit(pd.concat(train_dfs))

    #scale each df with scaler
    train_dfs = [pd.DataFrame(scaler.transform(df), columns=df.columns) for df in train_dfs]
    val_dfs = [pd.DataFrame(scaler.transform(df), columns=df.columns) for df in val_dfs]

    #prepare data for dataloader
    train_data = []
    for df in train_dfs:
        train_data += utils.make_dataloader_data(df, window_size=1)

    val_data = []
    for df in val_dfs:
        val_data += utils.make_dataloader_data(df, window_size=1)

    # pass data to dataloader
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
   
    # prepare directory for pretrained weights
    if not os.path.isdir('models'):
        os.mkdir('models')

    # prepare directory for history
    if not os.path.isdir('history'):
        os.mkdir('history')
    
    # define study parameters
    study_name = 'study_single_l1loss_sigmoid_small_blur'
    db_pth = '../00.db'
    storage = 'sqlite:///' + db_pth
    sampler = optuna.samplers.RandomSampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=20)

    with open(f'scaler_{study_name}.pickle', 'wb') as f:
                pickle.dump(scaler, f)

    # create/load and run study
    study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage, sampler=sampler, pruner=pruner, load_if_exists=True)
    '''
    study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)
    '''
    study.optimize(objective, n_trials=50)