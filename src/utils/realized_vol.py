import torch

def compute_rcov(returns,
                 rvol_proxy_name):
    T = returns.shape[0]
    k = returns.shape[1]

    rcov = torch.zeros(T, k, k)
    if rvol_proxy_name == "sample_cov":
        for t in range(returns.shape[0]):
            rts = torch.tensor(returns.iloc[t])

            rcov_t = torch.kron(rts, rts).reshape(k, k)
            rcov[t] = rcov_t
    else:
        raise Exception("Realized covariance proxy '{}' not implemented!!".format(rvol_proxy_name))

    return rcov

