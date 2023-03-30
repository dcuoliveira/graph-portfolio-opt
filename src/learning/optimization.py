import pandas as pd
from tqdm import tqdm

from utils.realized_vol import compute_rcov

def run_training_procedure(file,
                           model,
                           model_name,
                           rvol_proxy_name,
                           rvol_proxy_window,
                           init_steps,
                           prediction_steps,
                           embedd_init_steps):
    
    data = pd.read_excel(file)
    
    # 0. prepare dataset
    data.set_index("date", inplace=True)
    data = data.dropna()

    # 1. compute realized vol. proxy
    rcov_data = compute_rcov(data=data, rvol_proxy_name=rvol_proxy_name, rvol_proxy_window=rvol_proxy_window)
    
    # 2. use in-sample data to estimate graph embeddings
    embedd_data = data.iloc[0:embedd_init_steps]
    data = data.iloc[embedd_init_steps+1:]
    
    # 3. use graph structure and/or time series data to predict realized covariances
    rcovs = []
    preds = []
    forecast_errors = []
    for t in tqdm(range(init_steps, data.shape[0] - init_steps, prediction_steps),
                  desc="Running TSCV"):

        # get 1:t ts information
        train = data.iloc[:t]
        test = data.iloc[t:(t + prediction_steps)]

        # compute realized covariance

        # store realized covariance
        rcovs.append(None)

        # estimate model parameters

        # compute t+1 predictions

        # store t+1 realized covariance prediction
        preds.append(None)
        
        # compute forecast error
        
        # store forecast error
        forecast_errors.append(None)
