import pandas as pd
import numpy as np
import pickle
import os, sys
import torch
from torch import nn
from collections import OrderedDict
import scipy.signal as ss
import optuna
import json

from sklearn.preprocessing import MinMaxScaler


def load_trials(study_name, storage):
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [t for t in study.trials if t.value]
    trials = [t for t in sorted(trials, key=lambda x: x.value)]

    return trials


def load_trial(study_name, storage, rank=1):
    trials = load_trials(study_name, storage)
    trial = trials[rank - 1]

    return trial


def train_test_split(df, train_part=0.9):
    train_size = int(len(df) * train_part)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    return train_df, test_df


def preprocess_df(df, data_structure_file='data/data_structure.json', drop_null_cols=True, blur_loc_features=False, add_stop_feature=False):
    with open(data_structure_file, 'r') as f:
        data_structure = json.load(f)

    # select feature columns
    df = df[data_structure['columns']]

    # reset index
    df.reset_index(inplace=True, drop=True)

    # unify speed units
    df.way_maxspeed = df.way_maxspeed / 3.6

    # set all unknown categorical values to 'null' string
    df = set_categorical_cols_null_string(df, valid_categories=data_structure['categorical_values'])

    # encode all categorical values to one_hot encoding
    df = encode_categorical_one_hot(df, valid_categories=data_structure['categorical_values'])

    # ensure columns
    df = ensure_columns(df, valid_categories=data_structure['categorical_values'], feature_cols=data_structure['columns'])

    # reorder columns in DataFrame - ensure order of features
    df = reorder_columns(df)

    if add_stop_feature:
        df['node:car_stopped'] = 0
        df.loc[df.target_speed < 1.0, 'node:car_stopped'] = 1
        data_structure['categorical_values']['node:car_stopped'] = ['']

    # blur location features
    if blur_loc_features:
        df = blur_location_features(df, valid_categories=data_structure['categorical_values'])

    # drop complementary columns - columns with null or 0 value are complementary to the rest of the values
    if drop_null_cols:
        df.drop(columns=[cc for cc in df.columns.to_list() if cc.endswith('_null')] + ['start_stop_0'], inplace=True)

    assert len(df.columns.to_list()) in [43, 44, 50], "Expected 43 or 50 columns. {} found.".format(len(df.columns.to_list()))
    
    return df


def load_preprocess_drives(drive_pths, drop_null_cols=True, blur_loc_features=False, add_stop_feature=False):
    dfs = []
    for d in drive_pths:
        # load drive df
        df = pd.read_csv(d, index_col='index')
        # preprocess drive
        df = preprocess_df(
            df, 
            data_structure_file='../../data/data_structure.json',
            drop_null_cols=drop_null_cols,
            blur_loc_features=blur_loc_features,
            add_stop_feature=add_stop_feature
        )
        dfs.append(df)

    return dfs


def make_dataloader_data(df, window_size=1, padding='zeros'):
    if type(window_size) == tuple:
        bw = window_size[0]
        fw = window_size[1]
        if padding == 'circle':
            df = pd.concat([df[-bw:], df, df[:fw]])

        if padding == 'zeros':
            df = pd.concat([
                pd.DataFrame(np.zeros((bw, df.shape[1])), columns=df.columns),
                df, 
                pd.DataFrame(np.zeros((fw, df.shape[1])), columns=df.columns)
            ])

        else:
            raise ValueError('Padding method not supported.')

        y = df.target_speed.iloc[bw:-fw].to_numpy()
        X = df.drop(columns=['target_speed']).to_numpy()

        data = [(X[i:i+bw+fw+1], y[i]) for i in range(len((y)))]

    elif window_size == 1:
        y = df.target_speed.to_numpy()
        X = df.drop(columns=['target_speed']).to_numpy()

        data = [(X[i], y[i]) for i in range(len(y))]
        
    return data


def blur_location_features(df, valid_categories=None, window_size=201):
    window = ss.triang(window_size)
    pad_size = window_size // 2
    categorical_cols = valid_categories.keys()
    for cc in categorical_cols:
        for val in valid_categories[cc]:
            col_name = (str(cc) + '_' + str(val)).strip('_')
            if col_name.startswith('node:') or col_name.startswith('start_stop'):
                padded = np.pad(df[col_name], (pad_size), 'constant', constant_values=(0))
                df[col_name] = np.convolve(padded, window, 'valid')
                
    return df
    

def ensure_columns(df, valid_categories=None, feature_cols=None):
    numerical_cols = set(feature_cols) - set(valid_categories.keys())
    for nc in numerical_cols:
        if nc not in df:
            raise Exception("mandatory cols missing")
            
    categorical_cols = valid_categories.keys()
    for cc in categorical_cols:
        for val in valid_categories[cc]:
            col_name = str(cc) + '_' + str(val)
            # ensure all columns
            if col_name not in df:
                df[col_name] = 0
                
    return df


def set_categorical_cols_null_string(df, valid_categories=None):
    """All NaN values and values not specified in valid_categories are set to 'null' string"""
    categorical_cols = valid_categories.keys()
    for cc in categorical_cols:
        df[cc].fillna('null', inplace=True) # fill NaN values with 'null' string
        df.loc[~df[cc].isin(valid_categories[cc]), cc] = 'null' # set values not in valid_categories to 'null' string

    return df


def encode_categorical_one_hot(df, valid_categories=None):
    """Encodes all categorical columns in valid_categories to one_hot"""
    categorical_cols = valid_categories.keys()
    # encode categorical columns to one-hot    
    encoded_categoricals = pd.concat([pd.get_dummies(df[cc], prefix=cc) for cc in categorical_cols], axis=1)
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, encoded_categoricals], axis=1)
    
    return df


def reorder_columns(df):
    ordered_columns = sorted(df.columns.to_list()[1:])
    ordered_columns.insert(0, 'target_speed')
    df = df[ordered_columns]
    
    return df


def MAE(y, ycap):
    return np.mean(np.abs(y - ycap))


def predict_speed(X, trial, scaler, model, device):
    pred = model(torch.tensor(X).to(device))
    pred = pred.detach().cpu().numpy()
    if len(X[0].shape) == 1:
        fu = np.repeat(pred, X[0].shape[0] + 1, axis=1) # dimensions fix
    else:
        fu = np.repeat(pred, X[0].shape[1] + 1, axis=1) # dimensions problem with 1D input
    pred = scaler.inverse_transform(fu)[:, 0]
    
    return pred
