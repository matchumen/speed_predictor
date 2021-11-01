import sys, os
import utils as utils
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import optuna
import pickle
from eval_utils import evaluate_results


def test_model(
    experiment_dir=None, 
    study_name=None, 
    data_dir='data', 
    test_data_dir='test_dataset', 
    scaler_name=None, 
    model_fn=None,
    blur=False,
    stops=False
):
    # set target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load trial
    db_name = 'experimenty/00.db'
    storage = os.path.join("sqlite:///", db_name)
    trial = utils.load_trial(study_name, storage)

    # load test files paths
    test_files = [os.path.join(data_dir, test_data_dir, fn) for fn in os.listdir(os.path.join(data_dir, test_data_dir))]

    # load scaler
    with open(os.path.join(experiment_dir, scaler_name), 'rb') as file:
        scaler = pickle.load(file)

    if 'BW_SAMPLES' in trial.params.keys():
        window_size = (trial.params['BW_SAMPLES'], trial.params['FW_SAMPLES'])
    else:
        window_size = 1
    
    drives_results = []
    
    for tf in tqdm(test_files[:]):
        # load drive data
        df = pd.read_csv(tf, index_col='index')

        # preprocess drive data
        df = utils.preprocess_df(
            df, 
            data_structure_file='data/data_structure.json', 
            blur_loc_features=blur,
            add_stop_feature=stops
        )

        baseline_speed = np.array(df.speed_osrm)

        measured_speed = np.array(df.target_speed)

        # scale data
        df = pd.DataFrame(scaler.transform(df), columns=df.columns)

        # prepare data for model
        data = utils.make_dataloader_data(df, window_size=window_size)
        X = [d[0] for d in data]

        # add shape to trial
        if isinstance(window_size, int):
            trial.params['INPUT_SIZE'] = X[0].shape[0]
        else:
            trial.params['INPUT_SIZE'] = X[0].shape[0] * X[0].shape[1]
        # create model and load model weights
        model = model_fn(trial.params)
        model = model.double().to(device)
        model.load_state_dict(torch.load(
            os.path.join(experiment_dir, 'models', 'model_{:.6f}.pth'.format(trial.value)),
            map_location=torch.device(device))
        )
        model.eval()
   
        # predict speed
        predicted_speed = utils.predict_speed(X, trial, scaler, model, device)
        
        test_result = {
            'pth': tf,
            'measured': measured_speed,
            'baseline': baseline_speed,
            'predicted': predicted_speed
        }
        
        drives_results.append(test_result)

    test_results = evaluate_results(drives_results)

    return test_results
