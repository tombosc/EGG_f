import torch
import argparse
import os
from .data_readers import Data
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("interaction_file", type=str)
args = parser.parse_args()
exps_root = os.environ["EGG_EXPS_ROOT"]
data = torch.load(exps_root + '/' + args.interaction_file)
n_nec = Data.n_necessary_features(data.sender_input)
#  for n, inp, rinp, l, len_msg in zip(n_nec, data.sender_input, data.receiver_input,
#          data.labels, data.message_length):
#      print('---------')
#      print("necessary={}, label={}".format(n, l))
#      print(inp)
#      print(rinp)
plt.scatter(n_nec, data.message_length, alpha=0.05)
plt.show()
