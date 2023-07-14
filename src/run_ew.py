import os
import pandas as pd
import torch.utils.data as data
import torch
from tqdm import tqdm
import argparse
import json

from utils.dataset_utils import create_rolling_window_ts, timeseries_train_test_split
from loss_functions.SharpeLoss import SharpeLoss
from models.DLPO import DLPO
from data.NewETFs import NewETFs

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, help='epochs to be used on the training process', default=5)
parser.add_argument('-bs', '--batch_size', type=int, help='size of the batches for the time seriesd data', default=10)
parser.add_argument('-nti', '--num_timesteps_in', type=int, help='size of the lookback window for the time series data', default=50)
parser.add_argument('-nto', '--num_timesteps_out', type=int, help='size of the lookforward window to be predicted', default=1)
parser.add_argument('-ts', '--train_shuffle', type=bool, help='block shuffle train data', default=True)
parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model', default="dlpo")

if __name__ == "__main__":

    args = parser.parse_args()

    model_name = args.model_name

    # neural network hyperparameters
    input_size = 1426
    output_size = 1426
    hidden_size = 64
    num_layers = 1

    # optimization hyperparameters
    learning_rate = 1e-3

    # training hyperparameters
    device = torch.device('cpu')
    epochs = args.epochs
    batch_size = args.batch_size
    drop_last = True
    num_timesteps_in = args.num_timesteps_in
    num_timesteps_out = args.num_timesteps_out
    train_ratio = 0.5
    ascent = True
    fix_start=False
    train_shuffle = args.train_shuffle

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = NewETFs(use_last_data=True)
    prices = loader.prices.T
    returns = loader.returns.T
    features = loader.features
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).T    

    # call model

    # compute weights

    # save results

    results = {
        
        "model": model.state_dict(),
        "train_loss": train_loss_df,
        "eval_loss": eval_loss_df,
        "test_loss": avg_eval_loss,

        }

    output_path = os.path.join(os.path.dirname(__file__),
                                    "data",
                                    "outputs",
                                    model_name)

    # save args
    args_dict = vars(args)  
    with open(os.path.join(output_path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    # save results
    output_name = "{model_name}_{epochs}_{batch_size}_{num_timesteps_in}_{num_timesteps_out}.pt".format(model_name=model_name,
                                                                                                        epochs=epochs,
                                                                                                        batch_size=batch_size,
                                                                                                        num_timesteps_in=num_timesteps_in,
                                                                                                        num_timesteps_out=num_timesteps_out)
    torch.save(results, os.path.join(output_path, output_name))