import pandas as pd
from tqdm import tqdm

def run_training_procedure(file, model, init_steps, prediction_steps):
    
    data = pd.read_excel(file)
    
    # 1. use in-sample data to estimate graph structure
 
    # 2. use graph structure and/or time series data to predict realized covariances
    rcovs = []
    preds = []
    forecast_errors = []
    for t in tqdm(range(init_steps, data.shape[0] - init_steps, prediction_steps),
                  desc="Running TSCV"):

        # get 1:t ts information
        train = data[:t]
        test = data[t:(t + prediction_steps)]

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
