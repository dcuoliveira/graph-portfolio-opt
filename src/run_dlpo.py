import os
import pandas as pd
import torch.utils.data as data
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
                    default=5)

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

parser.add_argument('-ts',
                    '--train_shuffle',
                    type=bool,
                    help='block shuffle train data',
                    default=True)

parser.add_argument('-mn',
                    '--model_name',
                    type=str,
                    help='model name to be used for saving the model',
                    default="dlpo")

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
train_loader = data.DataLoader(data.TensorDataset(X_train, prices_train), shuffle=train_shuffle, batch_size=batch_size, drop_last=drop_last)
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
    train_loss = 0
    for X_batch, prices_batch in train_loader:
                
        # compute forward propagation
        weights_pred = model.forward(X_batch)

        # compute loss
        loss = lossfn(prices_batch, weights_pred, ascent=ascent)
        train_loss += loss.detach().item() * -1

        # compute gradients and backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # store average train loss
    avg_train_loss = train_loss / len(train_loader)
    train_loss_values.append(avg_train_loss)

    # evaluate model 
    model.eval()
    with torch.no_grad():
        eval_loss = 0
        for X_batch, prices_batch in val_loader:
            
            # compute forward propagation
            weights_pred = model.forward(X_batch)

            # compute loss
            loss = lossfn(prices_batch, weights_pred, ascent=ascent)
            eval_loss += loss.detach().item() * -1
    
        # store average evaluation loss
        avg_eval_loss = eval_loss / len(val_loader)
        eval_loss_values.append(avg_eval_loss)

    pbar.set_description("Epoch: %d, Train sharpe : %1.5f, Eval sharpe : %1.5f" % (epoch, avg_train_loss, avg_eval_loss))

train_loss_df = pd.DataFrame(train_loss_values, columns=["sharpe_ratio"])
eval_loss_df = pd.DataFrame(eval_loss_values, columns=["sharpe_ratio"])

with torch.no_grad():
    # compute train weight predictions
    weights_pred = model.forward(X_train)
    # select predictions
    weights_train_pred = weights_pred[:, -num_timesteps_out, :]

    # compute val weight predictions
    weights_pred = model.forward(X_val)
    # select predictions
    weights_eval_pred = weights_pred[:, -num_timesteps_out, :]

    # compute test weight predictions
    weights_pred = model.forward(X_test)
    # select predictions
    weights_test_pred = weights_pred[:, -num_timesteps_out, :]

results = {
    
    "model": model.state_dict(),
    "train_loss": train_loss_df,
    "eval_loss": eval_loss_df,
    "test_loss": avg_eval_loss,

    }

output_path = os.path.join(os.getcwd(),
                           "src,"
                           "data",
                           "outputs",
                           model_name)
output_name = "{model_name}_{epochs}_{batch_size}_{num_timesteps_in}_{num_timesteps_out}.pt".format(model_name=model_name,
                                                                                                    epochs=epochs,
                                                                                                    batch_size=batch_size,
                                                                                                    num_timesteps_in=num_timesteps_in,
                                                                                                    num_timesteps_out=num_timesteps_out)
torch.save(results, os.path.join(output_path, output_name))