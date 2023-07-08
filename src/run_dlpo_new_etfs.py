import os
import pandas as pd
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import argparse

from utils.dataset_utils import create_rolling_window_ts, timeseries_train_test_split
from loss_functions.SharpeLoss import SharpeLoss
from models.DLPO import DLPO
from data.NewETFs import NewETFs

parser = argparse.ArgumentParser()

parser.add_argument('-e',
                    '--epochs',
                    type=int,
                    help='epochs to be used on the training process',
                    default=10000)

parser.add_argument('-bs',
                    '--batch_size',
                    type=int,
                    help='size of the batches for the time seriesd data',
                    default=10)

parser.add_argument('-nti',
                    '--num_timesteps_in',
                    type=int,
                    help='size of the lookback window for the time series data',
                    default=50)

parser.add_argument('-nto',
                    '--num_timesteps_out',
                    type=int,
                    help='size of the lookforward window to be predicted',
                    default=1)

parser.add_argument('-mn',
                    '--model_name',
                    type=str,
                    help='model name to be used for saving the model',
                    default="dlpo")

args = parser.parse_args()

model_name = args.model_name

# neural network hyperparameters
input_size = 4 * 2
output_size = 4
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

# relevant paths
source_path = os.getcwd()
inputs_path = os.path.join(source_path, "data", "inputs")

# prepare dataset
loader = NewETFs(use_last_data=True)
prices = loader.y.T
features = loader.X
features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).T

# define train and test datasets
X_train, X_val, prices_train, prices_val = timeseries_train_test_split(features, prices, train_ratio=train_ratio)
X_val, X_test, prices_val, prices_test = timeseries_train_test_split(X_val, prices_val, train_ratio=0.5) 

X_train, prices_train = create_rolling_window_ts(features=X_train, 
                                                 target=prices_train,
                                                 num_timesteps_in=num_timesteps_in,
                                                 num_timesteps_out=num_timesteps_out,
                                                 fix_start=fix_start)
X_val, prices_val = create_rolling_window_ts(features=X_val, 
                                             target=prices_val,
                                             num_timesteps_in=num_timesteps_in,
                                             num_timesteps_out=num_timesteps_out,
                                             fix_start=fix_start)
X_test, prices_test = create_rolling_window_ts(features=X_test, 
                                               target=prices_test,
                                               num_timesteps_in=num_timesteps_in,
                                               num_timesteps_out=num_timesteps_out,
                                               fix_start=fix_start)

# define data loaders
train_loader = data.DataLoader(data.TensorDataset(X_train, prices_train), shuffle=False, batch_size=batch_size, drop_last=drop_last)
val_loader = data.DataLoader(data.TensorDataset(X_val, prices_val), shuffle=False, batch_size=batch_size, drop_last=drop_last)
test_loader = data.DataLoader(data.TensorDataset(X_test, prices_test), shuffle=False, batch_size=batch_size, drop_last=drop_last)

# (1) model
model = DLPO(input_size=input_size,
             output_size=output_size,
             hidden_size=hidden_size,
             num_layers=num_layers,
             batch_first=True,
             num_timesteps_out=num_timesteps_out).to(device)

# (2) loss fucntion
lossfn = SharpeLoss()

# (3) optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# (4) training procedure
train_loss_values = eval_loss_values = test_loss_values = []
pbar = tqdm(range(epochs + 1), total=(epochs + 1))
for epoch in pbar:

    # train model
    model.train()
    for X_batch, prices_batch in train_loader:
                
        # compute forward propagation
        # NOTE - despite num_timesteps_out=1, the predictions are being made on the batch_size(=10) dimension. Need to fix that.
        weights_pred = model.forward(X_batch)

        # compute loss
        loss = lossfn(prices_batch, weights_pred, ascent=ascent)

        # compute gradients and backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_values.append(loss.detach().item() * -1)

    train_loss = (loss.detach().item() * -1)

    # evaluate model 
    model.eval()
    with torch.no_grad():
        for X_batch, prices_batch in test_loader:
            # compute forward propagation
            weights_pred = model.forward(X_batch)

            # compute loss
            loss = lossfn(prices_batch, weights_pred, ascent=ascent)
            eval_loss_values.append(loss.detach().item() * -1)

    eval_loss = (loss.detach().item() * -1)

    pbar.set_description("Epoch: %d, Train sharpe : %1.5f, Eval sharpe : %1.5f" % (epoch, train_loss, eval_loss))

train_loss_df = pd.DataFrame(train_loss_values, columns=["sharpe_ratio"])
eval_loss_df = pd.DataFrame(eval_loss_values, columns=["sharpe_ratio"])

model.eval()
pbar = tqdm(enumerate(test_loader), total=len(test_loader))
for i, (X_batch, prices_batch) in pbar:
    
    optimizer.zero_grad()
    
    # compute forward propagation
    weights_pred = model.forward(X_batch)

    # compute loss
    loss = lossfn(prices_batch, weights_pred, ascent=True)
    
    # compute gradients and backpropagate
    loss.backward()
    optimizer.step()
    pbar.set_description("Test sharpe : %1.5f" % (loss.item() * -1))

    test_loss_values.append(loss.detach().item() * -1)

test_loss_df = pd.DataFrame(test_loss_values, columns=["sharpe_ratio"])

results = {
    
    "model": model.state_dict(),
    "train_loss": train_loss_df,
    "eval_loss": eval_loss_df,
    "test_loss": test_loss_df,

    }

output_path = os.path.join(os.getcwd(),
                           "data",
                           "outputs",
                           model_name)
output_name = "{model_name}_{epochs}_{batch_size}_{num_timesteps_in}_{num_timesteps_out}.pt".format(model_name=model_name,
                                                                                                    epochs=epochs,
                                                                                                    batch_size=batch_size,
                                                                                                    num_timesteps_in=num_timesteps_in,
                                                                                                    num_timesteps_out=num_timesteps_out)
torch.save(results, os.path.join(output_path, output_name))