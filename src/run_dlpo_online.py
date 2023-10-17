import os
import pandas as pd
import torch.utils.data as data
import torch
from tqdm import tqdm
import argparse

from utils.dataset_utils import create_online_rolling_window_ts, timeseries_train_test_split_online
from loss_functions.SharpeLoss import SharpeLoss
from models.DLPO import DLPO
from data.CRSPSimple import CRSPSimple
from utils.conn_data import save_result_in_blocks
from utils.dataset_utils import check_bool

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, help='epochs to be used on the training process', default=100)
parser.add_argument('-bs', '--batch_size', type=int, help='size of the batches for the time seriesd data', default=90)
parser.add_argument('-nti', '--num_timesteps_in', type=int, help='size of the lookback window for the time series data', default=252 * 3)
parser.add_argument('-nto', '--num_timesteps_out', type=int, help='size of the lookforward window to be predicted', default=1)
parser.add_argument('-ts', '--train_shuffle', type=str, help='block shuffle train data', default="False")
parser.add_argument('--train_ratio', type=int, default=0.7, help='ratio of the data to consider as training')
parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model', default="dlpo")
parser.add_argument('--use_small_data', type=str, help='use small sample stocks data', default="True")
parser.add_argument('-usd', '--use_sample_data', type=str, help='use sample stocks data', default="False")
parser.add_argument('-ay', '--all_years', type=str, help='use all years to build dataset', default="False")
parser.add_argument('-lo', '--long_only', type=str, help='consider long only constraint on the optimization', default="False")

if __name__ == "__main__":

    args = parser.parse_args()

    model_name = args.model_name

    # optimization hyperparameters
    learning_rate = 1e-3

    # training hyperparameters
    device = torch.device('cpu')
    epochs = args.epochs
    batch_size = args.batch_size
    drop_last = True
    num_timesteps_in = args.num_timesteps_in
    num_timesteps_out = args.num_timesteps_out
    ascent = True
    fix_start=False
    train_ratio = args.train_ratio
    train_shuffle = check_bool(args.train_shuffle)
    use_small_data = check_bool(args.use_small_data)
    use_sample_data = check_bool(args.use_sample_data)
    all_years = check_bool(args.all_years)
    long_only = check_bool(args.long_only)

    print("Running script with the following parameters: model_name: {}, use_sample_data: {}, all_years: {}, long_only: {}".format(model_name, use_sample_data, all_years, long_only))

    model_name = "{model_name}_lo".format(model_name=model_name) if long_only else "{model_name}_ls".format(model_name=model_name)
    model_name = "{}_sample".format(model_name) if use_sample_data else model_name
    model_name = "{}_shuffle".format(model_name) if train_shuffle else model_name

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = CRSPSimple(use_sample_data=use_sample_data, all_years=all_years)
    returns = loader.returns.T
    features = loader.features
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).T  

    X_steps, returns_steps = create_online_rolling_window_ts(features=features, 
                                                             target=returns,
                                                             num_timesteps_in=num_timesteps_in,
                                                             num_timesteps_out=num_timesteps_out,
                                                             fix_start=fix_start,
                                                             drop_last=drop_last)

    # neural network hyperparameters
    input_size = returns.shape[1]
    output_size = returns.shape[1]
    hidden_size = input_size * 4
    num_layers = 1

    # (1) model
    model = DLPO(input_size=input_size,
                 output_size=output_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 batch_first=False,
                 num_timesteps_out=num_timesteps_out).to(device)

    # (2) loss fucntion
    lossfn = SharpeLoss()

    # (3) optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # (4) training/validation + oos testing
    test_weights = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    test_returns = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    train_loss = torch.zeros((X_steps.shape[0], 1))
    test_loss = torch.zeros((X_steps.shape[0], 1))

    pbar = tqdm(range(X_steps.shape[0]-1), total=(X_steps.shape[0] + 1))
    for step in pbar:
        X_t = X_steps[step, :, :]
        returns_t1 = returns_steps[step, :, :]

        X_train_t, X_test_t, returns_train_t1, returns_test_t1 = timeseries_train_test_split_online(X=X_t,
                                                                                                    y=returns_t1,
                                                                                                    train_ratio=train_ratio)
        
        train_loader = data.DataLoader(data.TensorDataset(X_train_t, returns_train_t1),
                                       shuffle=train_shuffle,
                                       batch_size=batch_size,
                                       drop_last=drop_last)

        train_loss_vals = 0
        for epoch in range(epochs):

            # train/validate model
            model.train()
            for X_batch, returns_batch in train_loader:
                        
                # compute forward propagation
                weights_t1 = model.forward(X_batch[None, :, :], long_only=long_only)

                # compute loss
                loss = lossfn(returns=returns_batch[-num_timesteps_out:], weights=weights_t1, ascent=ascent)
                train_loss_vals += loss.detach().item() * -1

                # compute gradients and backpropagate
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss_vals = train_loss_vals / (epochs * len(train_loader))

        # oos test model 
        model.eval()
        with torch.no_grad():

            # compute forward propagation
            weights_t1 = model.forward(X_test_t[None, :, :], long_only=long_only)

            # compute loss
            loss = lossfn(returns=returns_test_t1[-num_timesteps_out:], weights=weights_t1, ascent=ascent)
            eval_loss_vals = loss.detach().item() * -1

            # save results
            test_weights[step, :, :] = weights_t1
            test_returns[step, :, :] = returns_test_t1[-num_timesteps_out:]

        train_loss[step, :] = avg_train_loss_vals
        test_loss[step, :] = eval_loss_vals

        pbar.set_description("Steps: %d, Train sharpe : %1.5f, Test sharpe : %1.5f" % (step, avg_train_loss_vals, eval_loss_vals))

    if test_weights.dim() == 3:
        weights = test_weights.reshape(test_weights.shape[0] * test_weights.shape[1], test_weights.shape[2])
    else:
        weights = test_weights

    if test_returns.dim() == 3:
        returns = test_returns.reshape(test_weights.shape[0] * test_weights.shape[1], test_weights.shape[2])
    else:
        returns = test_returns

    # (4) save results
    returns_df = pd.DataFrame(returns.numpy(), index=loader.index[(num_timesteps_in + num_timesteps_out):], columns=loader.columns)
    weights_df = pd.DataFrame(weights.numpy(), index=loader.index[(num_timesteps_in + num_timesteps_out):], columns=loader.columns)
    
    melt_returns_df = returns_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "returns"})
    melt_weights_df = weights_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "weights"})
    summary_df = pd.merge(melt_returns_df, melt_weights_df, on=["date", "ticker"], how="inner")

    results = {
        
        "model": model.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "returns": returns_df,
        "weights": weights_df,
        "summary": summary_df

        }

    output_path = os.path.join(os.path.dirname(__file__),
                                    "data",
                                    "outputs",
                                    model_name)
    
    save_result_in_blocks(results=results, args=args, path=output_path)
