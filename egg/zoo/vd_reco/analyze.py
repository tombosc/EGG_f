# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.distributions import Categorical
from functools import partial

import egg.core as core
from egg.core.smorms3 import SMORMS3
from egg.core import EarlyStopperNoImprovement
from egg.core.baselines import MeanBaseline, SenderLikeBaseline
from .archs_protoroles import Hyperparameters, load_game
from .train_vl import loss_objs
#  from .data_proto import init_data as init_data_proto
from .data_proto import Data as DataProto, init_data as init_data_proto
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from simple_parsing import ArgumentParser
from egg.core import Trainer
import egg.core.util as util
from egg.core.distributed import not_distributed_context


def get_model_data(params):
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    #  parser.add_argument('train_interactions', type=str)
    args = parser.parse_args(params)
    dirname = os.path.dirname(args.checkpoint)
    dataset_json = os.path.join(dirname, 'data.json')
    with open(dataset_json, 'r') as f:
        json_data = json.load(f)
        dataset_name = json_data["dataset"]
    checkpoint = torch.load(args.checkpoint)
    hp_json = os.path.join(dirname, 'hp.json')
    if dataset_name == 'proto':
        data_cfg = DataProto.Settings.load(dataset_json)
        dataset, train_data, valid_data, test_data = init_data_proto(data_cfg, 0, 128)
        model = load_game(Hyperparameters.load(hp_json), loss_objs)
        model.load_state_dict(checkpoint.model_state_dict)
    else:
        raise ValueError('Unknown dataset', dataset_name)
    #  train_interactions = torch.load(args.train_interactions)
    return dirname, model, dataset, train_data, valid_data, test_data


def count_argument_positions(interactions):
    arguments_counts = Counter()
    #  position_counts_per_arg = defaultdict(Counter)
    for i, m in enumerate(interactions.message.argmax(2)):
        to_send = interactions.aux["sender_input_to_send"][i]
        for j, v in enumerate(to_send[1:]):
            if v.item() == 1:
                arg = tuple(interactions.sender_input[i][j].tolist())
                arguments_counts[arg] += 1
                #  position_counts_per_arg[arg][j] += 1
    return arguments_counts


def main(params):
    dirname, model, dataset, train_data, valid_data, test_data = get_model_data(params)
    util.common_opts = argparse.Namespace()
    util.no_distributed = True
    util.common_opts.preemptable = False
    util.common_opts.validation_freq = 0
    util.common_opts.update_freq = 0
    util.common_opts.checkpoint_dir = None
    util.common_opts.checkpoint_freq = 0
    util.common_opts.checkpoint_best = ""
    util.common_opts.tensorboard = False
    util.common_opts.fp16 = False
    util.common_opts.load_from_checkpoint = None
    util.common_opts.distributed_context = not_distributed_context()

    evaluator = Trainer(model, None, train_data, valid_data, 'cpu', None, None, False)
    loss_valid, I = evaluator.eval()
    score_path_json = os.path.join(dirname, 'best_scores.json')
    with open(score_path_json, 'r') as f:
        best_score = json.load(f)['best_score']
    print("Best score vs eval", loss_valid, best_score)
    exit()
    # TODO fix the rest

    j = 0
    marker = 99
    roleset_permutations = dataset.roleset_permutations
    for argument, count in arg_counts.most_common(30):
        print("Arg #{}: {} ({})".format(j, argument, count))
        j += 1
        per_rolesets = defaultdict(list)
        for i, loss_objs in enumerate(I.aux['loss_objs']):
            loss_objs = loss_objs.item()
            to_send = I.aux["sender_input_to_send"][i][1:].int()
            for k, v in enumerate(to_send):
                if v == 1:
                    arg = tuple(I.sender_input[i][k].tolist())
                    if arg == argument:
                        to_send[k] = marker
            if marker not in to_send or loss_objs > 4:
                continue
            roleset = int(I.aux['roleset'][i].item())
            m = I.message[i].argmax(1)
            perm_to_send = tuple(to_send[roleset_permutations[roleset]].tolist())
            per_rolesets[roleset].append(
                (m, tuple(to_send.tolist()), perm_to_send, loss_objs))
        for roleset, data in per_rolesets.items():
            if len(data) < 2:
                continue
            sorted_data = sorted(data, key=lambda e: e[1])
            print("Roleset", roleset)
            for m, to_send, P_to_send, loss in sorted_data:
                print(m, to_send, P_to_send, loss)
        print("END")



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
