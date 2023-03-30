def compute_rcov(data,
                 rvol_proxy_name,
                 rvol_proxy_window):
    
    if rvol_proxy_name == "sample_cov":
        data = data.rolling(window=rvol_proxy_window).cov()