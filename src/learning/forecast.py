import torch

def forecast(returns,
             model,
             model_name,
             prediction_steps):
    
    if model_name == "dcc_garch":
        model_fit = model.fit(returns)
        model_out = model.predict(prediction_steps)

        rcov_tp1 = torch.tensor(model_out["cov"])
    else:
        raise Exception("Forcasting method '{}' not implemented".format(model_name))

    return rcov_tp1
    
