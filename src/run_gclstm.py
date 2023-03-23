import os
import argparse

from learning.optimization import run_training_procedure
from data.dataset_names import ds_names
from models.model_names import model_names

source_path = os.path.dirname(os.__file__)
inputs_path = os.path.join(source_path, "inputs")

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, default="etfs", help='Dataset name.')
parser.add_argument('--model_name', type=str, default="gclstm", help='Model name.')
parser.add_argument('--init_steps', type=int, default=252, help='Model name.')
parser.add_argument('--prediction_steps', type=int, default=1, help='Model name.')

if __name__ == "__main__":
    args = parser.parse_args()
    results = run_training_procedure(file=os.path.join(inputs_path, ds_names[args.ds_name]),
                                     model=model_names[args.model_name],
                                     model_name=args.model_name,
                                     init_steps=args.init_steps,
                                     prediction_steps=args.prediction_steps)
