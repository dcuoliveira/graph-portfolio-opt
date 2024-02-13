import os
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm
from copy import copy

from models.MVO import MVO
from data.CRSPLoader import CRSPLoader
from utils.dataset_utils import check_bool
from loss_functions.SharpeLoss import SharpeLoss
from utils.conn_data import save_result_in_blocks

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model', default="mvo")
parser.add_argument('-wl', '--window_length', type=int, help='size of the lookback window for the time series data', default=50)
parser.add_argument('-sl', '--step_length', type=int, help='size of the lookforward window to be predicted', default=1)
parser.add_argument('-lo', '--long_only', type=str, help='consider long only constraint on the optimization', default="False")
parser.add_argument('-meane', '--mean_estimator', type=str, help='name of the estimator to be used for the expected returns', default="mle")
parser.add_argument('-cove', '--covariance_estimator', type=str, help='name of the estimator to be used for the covariance of the returns', default="mle")

if __name__ == "__main__":

    args = parser.parse_args()

    args.model = copy(args.model_name)

    model_name = args.model_name
    train_ratio = 0.6
    window_length = args.window_length
    step_length = args.step_length
    fix_start = False
    drop_last = True
    long_only = check_bool(args.long_only)
    mean_estimator = args.mean_estimator
    covariance_estimator = args.covariance_estimator

    etf_tickers = ['SPY', 'XLF', 'XLB', 'XLK', 'XLV', 'XLI', 'XLU', 'XLY', 'XLP', 'XLE']
    num_feat_names = ['pvCLCL']
    cat_feat_names = []
    
    # add tag for long only or long-short portfolios
    model_name = "{model_name}_lo".format(model_name=model_name) if long_only else "{model_name}_ls".format(model_name=model_name)

    # add mean estimator tag to name
    model_name = "{model_name}_{mean_estimator}".format(model_name=model_name, mean_estimator=mean_estimator)

    # add covariance estimator tag to name
    model_name = "{model_name}_{covariance_estimator}".format(model_name=model_name, covariance_estimator=covariance_estimator)
    
    args.model_name = model_name

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = CRSPLoader(load_data=True,
                        load_path=os.path.join(os.getcwd(), "src", "data", "inputs"),
                        load_edge_data=True,
                        num_feat_names=num_feat_names,
                        cat_feat_names=cat_feat_names)
    loader._update_ticker_index(ticker_list=etf_tickers)
    dataset = loader.get_dataset(data=loader.select_tickers(tickers=etf_tickers),
                                    window_length=window_length,
                                    step_length=step_length)

    # (1) call model
    model = MVO(mean_estimator=mean_estimator,
                covariance_estimator=covariance_estimator)

    # (2) loss fucntion
    lossfn = SharpeLoss()

    # (3) training/validation + oos testing
    test_weights = torch.zeros((len(loader.dates), loader.num_features, loader.num_tickers))
    returns = torch.zeros((len(loader.dates), loader.num_features, loader.num_tickers))
    test_loss = torch.zeros((len(loader.dates), step_length))

    pbar = tqdm(enumerate(dataset), total=dataset.get_num_batches())
    for step, batch in pbar:

         # sanity checks
        if batch.x.shape[0] != 1:
            raise ValueError("Batch size should be 1")
        
        if batch.x.shape[2] != 1:
            raise ValueError("Number of features should be 1")

        # select features and target
        features, target = batch.x[0, :, 0, :].T, batch.y.T
        returns_t1 = target[-2:-1, :] if target.shape[0] > 1 else target

        if target.shape[0] == 0:
            continue

        # compute weights
        weights = model.forward(returns=features, num_timesteps_out=step_length, long_only=long_only)

        loss = lossfn(weights=weights, returns=returns_t1)

        # save outputs
        test_weights[step, :, :] = weights
        returns[step, :, :] = returns_t1
        test_loss[step, :] = loss.item()

        pbar.set_description("Steps: %d, Test sharpe : %1.5f" % (step, loss.item()))

    # (4) save results
    returns_df = pd.DataFrame(returns[:, 0, :].numpy(), index=loader.dates, columns=etf_tickers)
    weights_df = pd.DataFrame(test_weights[:, 0, :].numpy(), index=loader.dates, columns=etf_tickers)
    
    melt_returns_df = returns_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "returns"})
    melt_weights_df = weights_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "weights"})
    summary_df = pd.merge(melt_returns_df, melt_weights_df, on=["date", "ticker"], how="inner")

    results = {
        
        "model": model,
        "means": None,
        "covs": None,
        "train_loss": None,
        "eval_loss": None,
        "test_loss": test_loss,
        "returns": returns_df,
        "weights": weights_df,
        "summary": summary_df

        }

    output_path = os.path.join(os.path.dirname(__file__),
                               "data",
                               "outputs",
                               model_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_result_in_blocks(results=results, args=args, path=output_path)