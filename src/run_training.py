import os
import argparse

from learning.optimization import run_training_procedure
from data.dataset_names import ds_names
from models.model_names import model_names
from utils.conn_data import save_pickle
from settings import OUTPUTS_PATH

source_path = os.path.dirname(__file__)
inputs_path = os.path.join(source_path, "data", "inputs")

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, default="etfs", help='Dataset name.')
parser.add_argument('--model_name', type=str, default="dcc_garch", help='Model name.')
parser.add_argument('--rvol_proxy_name', type=str, default="sample_cov", help='Realized vol. proxy name to be computed.')
parser.add_argument('--init_steps', type=int, default=252, help='Init steps to estimate before predicting.')
parser.add_argument('--prediction_steps', type=int, default=1, help='Steps ahead to predict.')
parser.add_argument('--embedd_init_steps', type=int, default=252, help='Steps to estimate graph embeddings.')

if __name__ == "__main__":
    args = parser.parse_args()
    results = run_training_procedure(file=os.path.join(inputs_path, ds_names[args.ds_name]),
                                     model=model_names[args.model_name],
                                     rvol_proxy_name=args.rvol_proxy_name,
                                     model_name=args.model_name,
                                     init_steps=args.init_steps,
                                     prediction_steps=args.prediction_steps,
                                     embedd_init_steps=args.embedd_init_steps)
    
    save_pickle(path=os.path.join(OUTPUTS_PATH, args.model_name,  "training_results.pickle"),
                obj=results)
