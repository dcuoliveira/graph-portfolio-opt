import os
import json
import pickle
import pandas as pd

def save_result_in_blocks(results, args, path):

    if not os.path.exists(path):
        os.makedirs(path)

    # save args
    args_dict = vars(args)  
    with open(os.path.join(path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    years = list(pd.Series([dtref.year for dtref in results["summary"]["date"]]).unique())

    for y in years:
        
        tmp_results = {

            "train_loss": None,
            "eval_loss": None,
            "test_loss": None,
            "returns": results["returns"].loc[results["returns"].index.year == y],
            "weights": results["weights"].loc[results["weights"].index.year == y],
            "summary": results["summary"].loc[results["summary"]["date"].dt.year == y]

        } 

        # save results
        save_pickle(obj=results, path=os.path.join(path, "results_{}.pickle".format(y)))
        tmp_results["summary"].to_csv(os.path.join(path, "summary_{}.csv".format(y)), index=False)

    if results["test_loss"] is not None:
        save_pickle(obj={"test_loss": results["test_loss"]}, path=os.path.join(path, "test_loss.pickle")) 

    save_pickle(obj={"model": results["model"]}, path=os.path.join(path, "model.pickle"))

def save_pickle(path: str,
                obj: dict):

    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):
    file = open(path, 'rb')
    target_dict = pickle.load(file)

    return target_dict