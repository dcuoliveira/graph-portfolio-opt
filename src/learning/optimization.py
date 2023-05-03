import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from learning.forecast import forecast
from utils.dataset_utils import concatenate_prices_returns

def run_training_procedure(file,
                           model,
                           model_name,
                           init_steps,
                           prediction_steps):
    
    prices = pd.read_excel(file)
    
    # 0. prepare dataset
    prices.set_index("date", inplace=True)
    returns = np.log(prices).diff().dropna()
    prices = prices.loc[returns.index]
    all_df = concatenate_prices_returns(prices=prices, returns=returns)
    
    # 1. use time series of prices and returns to predict portfolio weights
    T = (all_df.shape[0] - init_steps)
    k = all_df.shape[1]
    idx = 0

    preds = losses = torch.zeros(T, k, k)
    for t in tqdm(range(init_steps, all_df.shape[0] - init_steps, prediction_steps), desc="Running forecast procedure"):

        # get 1:t ts information
        train = all_df.iloc[:t]

        # estimate model parameters and compute t+1 predictions
        rcov_tp1 = forecast(returns=train,
                             model=model,
                            model_name=model_name,
                            prediction_steps=prediction_steps)

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
