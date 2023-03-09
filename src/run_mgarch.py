import os

from models.DCC_MGARCH import DCC_MGARCH
from learning.optimization import run_training_procedure

source_path = os.path.dirname(os.__file__)
inputs_path = os.path.join(source_path, "inputs")
ds_name = "etfs-zhang-zohren-roberts.xlsx"
model = DCC_MGARCH
init_steps = 252
prediction_steps = 1

if __name__ == "__main__":
    results = run_training_procedure(file=os.path.join(inputs_path, ds_name),
                                     model=model,
                                     init_steps=init_steps,
                                     prediction_steps=prediction_steps)
