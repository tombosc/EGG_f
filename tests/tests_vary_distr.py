import importlib
import tempfile
import os
import pathlib
import shutil
import sys
import torch
from egg.zoo.vary_distr.data_readers import Data, GeneratedData


def run_game(game, params):
    #  dev_null_file = open(os.devnull, "w")
    #  old_stdout, sys.stdout = sys.stdout, dev_null_file

    game = importlib.import_module(game)
    params_as_list = [f"--{k}={v}" for k, v in params.items()]
    game.main(params_as_list)

    #  sys.std_out = old_stdout

data_example = b"""5 3
1 0 0 0 2 . 1 0 1 0 1 . 1
1 0 1 1 0 . 2 2 1 0 0 . 0 1 2 0 2 . 0
1 0 1 1 0 . 2 2 1 0 0 . 0 1 2 0 2 . 0 1 2 0 2 . 3
1 0 0 0 2 . 1 0 1 0 1 . 1"""

def test_game():
    with tempfile.NamedTemporaryFile() as f:
        f.write(data_example)
        f.seek(0) 
        run_game(
            "egg.zoo.vary_distr.play",
            dict(vocab_size=3, n_epoch=1, max_len=2),
        )


# TODO do we really need Data? Or GeneratedData is enough
#  def test_data():
#      data = Data.from_str(data_example, one_hot=False)
#      sender_input, label, receiver_input = data[0]
#      print("sender_input: {}\n, labels: {}\n, receiver: {}\n".format(
#          sender_input, label, receiver_input,
#      ))
#      for sender_input, label, receiver_input in data:
#          print("Y")
#          assert(torch.all(sender_input[0] == receiver_input[label]))
#      assert(data.get_n_features() == 5)
#  assert(False)

def test_data_generation():
    N = 29
    max_value = 6
    n_features = 11
    data = GeneratedData(N, max_value, 1, 10, n_features, seed=47)
    assert(len(data) == N)
    for sender_input, label, receiver_input in data:
        assert(torch.all(sender_input < max_value + 1))
        assert(torch.all(0 <= sender_input))
        assert(sender_input.size(1) == n_features)
        assert(torch.all(sender_input[0] == receiver_input[label]))
    assert(data.get_n_features() == n_features)
