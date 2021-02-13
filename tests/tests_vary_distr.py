import importlib
import tempfile
import os
import pathlib
import shutil
import sys
import torch
from egg.zoo.vary_distr.data_readers import Data
from collections import Counter


def run_game(game, params):
    #  dev_null_file = open(os.devnull, "w")
    #  old_stdout, sys.stdout = sys.stdout, dev_null_file

    game = importlib.import_module(game)
    params_as_list = [f"--{k}={v}" for k, v in params.items()]
    game.main(params_as_list)

    #  sys.std_out = old_stdout

def test_data_generation():
    c = Data.Config(
        n_examples = 29,
        max_value = 6,
        n_features = 3,
        min_distractors = 1,
        max_distractors = 10,
        seed = 47
    )
    data = Data(c)
    assert(len(data) == c.n_examples)
    for sender_input, label, receiver_input in data:
        assert(torch.all(sender_input < c.max_value + 1))
        assert(torch.all(0 <= sender_input))
        assert(sender_input.size(1) == c.n_features)
        # most important test: whether the target is well-positionned
        assert(torch.all(sender_input[0] == receiver_input[label]))
    assert(data.get_n_features() == c.n_features)

def test_n_necessary_features():
    inp = torch.tensor([
        [[1, 2, 1],
         [2, 2, 1],
         [0, 0, 0]],
        [[1, 2, 1],
         [2, 2, 1],
         [1, 2, 2]],

    ])
    n_necessary = Data.n_necessary_features(inp)
    print(n_necessary)
    assert(n_necessary[0] == 1)
    assert(n_necessary[1] == 2)
    inp = torch.tensor([
        [[1, 2, 1, 3],
         [1, 2, 1, 2],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 2, 1, 3],
         [1, 2, 1, 2],
         [1, 2, 2, 3],
         [3, 2, 1, 3]],
    ])
    n_necessary = Data.n_necessary_features(inp)
    assert(n_necessary[0] == 1)
    assert(n_necessary[1] == 3)
    data = Data(Data.Config())
    features, n_necessary_features = data.generate_example()

def test_count_necess():
    c = Data.Config(
        n_examples = 1000,
        max_value = 5, 
        n_features = 3,
        min_distractors = 2,
        max_distractors = 10,
        seed = 2
    )
    data = Data(c)
    counts = Counter()
    for i in range(1000):
        sender_input = data[i][0].unsqueeze(0)
        #  print(sender_input)
        counts[Data.n_necessary_features(sender_input).item()] += 1
    print(counts)

