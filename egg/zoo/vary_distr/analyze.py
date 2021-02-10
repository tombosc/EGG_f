import torch
import argparse
import os
import json
from .data_readers import Data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("interaction_file", type=str)
args = parser.parse_args()
exps_root = os.environ["EGG_EXPS_ROOT"]
data = torch.load(exps_root + '/' + args.interaction_file)
n_nec = Data.n_necessary_features(data.sender_input)

n_nec = n_nec.float().tolist()
lens = data.message_length.float().tolist()
n_nec_unique = sorted(list(set(n_nec)))
for i, n in enumerate(n_nec_unique):
    X = np.asarray([l for l, n_ in zip(lens, n_nec) if n_ == n])
    results = {
        'n_necessary': n,
        'n_datapoints': len(X),
        'mean_data': np.mean(X),
        'std_data': np.std(X),
    }
    print(json.dumps(results))
    
for m in data.message.float().tolist():
    print(m)
