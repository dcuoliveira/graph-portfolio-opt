import torch

from utils.realized_vol import compute_rcov

def forecast(returns,
             model,
             model_name,
             prediction_steps,
             ovrd=None):
    
    if model_name == "dcc_garch":
        model_fit = model.fit(returns)
        model_out = model.predict(prediction_steps)
        rcov_tp1 = torch.tensor(model_out["cov"])
    elif model_name == "rw":
        model_fit = compute_rcov(returns=returns, rvol_proxy_name=ovrd)
        rcov_tp1 = model_fit[-1]
    else:
        raise Exception("Forcasting method '{}' not implemented".format(model_name))

    return rcov_tp1
    
