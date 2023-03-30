import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from utils.realized_vol import compute_rcov
from learning.forecast import forecast

def run_training_procedure(file,
                           model,
                           model_name,
                           rvol_proxy_name,
                           init_steps,
                           prediction_steps,
                           embedd_init_steps):
    
    prices = pd.read_excel(file)
    
    # 0. prepare dataset
    prices.set_index("date", inplace=True)
    prices = prices
    returns = np.log(prices).diff().dropna()

    # 1. use in-sample data to estimate graph embeddings
    embedd_returns = returns.iloc[0:embedd_init_steps]
    returns = returns.iloc[embedd_init_steps+1:]

    # 2. compute realized vol. proxy
    rcovs_true = compute_rcov(returns=returns, rvol_proxy_name=rvol_proxy_name)
    
    # 3. use graph structure and/or time series data to predict realized covariances
    T = (returns.shape[0] - init_steps)
    k = returns.shape[1]
    idx = 0

    preds = losses = torch.zeros(T, k, k)
    for t in tqdm(range(init_steps, returns.shape[0] - init_steps, prediction_steps), desc="Running forecast procedure"):

        # get 1:t ts information
        train = returns.iloc[:t]

        # estimate model parameters and compute t+1 predictions
        rcov_tp1 = forecast(returns=train, model=model,
                            model_name=model_name,
                            prediction_steps=prediction_steps,
                            ovrd=rvol_proxy_name if model_name == "rw" else None)

        # store t+1 realized covariance prediction
        preds[idx] = rcov_tp1
        
        # compute forecast error
        loss = F.mse_loss(input=rcov_tp1, target=rcovs_true[t])
        
        # store forecast error
        losses[idx] = loss

        idx += 1
    
    results = {
        
        "rcov": rcovs_true,
        "predictions": preds,
        "loss": losses,

        }
    
    return results
