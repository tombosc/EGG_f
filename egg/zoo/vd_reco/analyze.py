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
from .archs_protoroles import (Sender, TransformerSenderGS, Receiver,
        SenderReceiverTransformerGS)
from .data_proto import Data as DataProto
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from simple_parsing import ArgumentParser


def get_model_data(params):
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('dataset_json', type=str)
    parser.add_argument('train_interactions', type=str)
    args = parser.parse_args(params)
    with open(args.dataset_json, 'r') as f:
        json_data = json.load(f)
        dataset_name = json_data["dataset"]
    if dataset_name == 'proto':
        data_cfg = DataProto.Settings.load(args.dataset_json)
        # TODO now data is ignored, only interactions matter
        data = DataProto(seed=data_cfg.dataset_seed, augment=data_cfg.augment,
                         shuffle_roles=data_cfg.shuffle_roles)
    else:
        raise ValueError('Unknown dataset', dataset_name)
    ratios = (data_cfg.train_ratio, data_cfg.valid_ratio,
              1 - (data_cfg.train_ratio + data_cfg.valid_ratio))
    data_gen = torch.Generator()
    #  train_data, _, test_data = data.random_split(ratios, data_gen)
    train_data, test_data = None, None
    #  model = torch.load(args.checkpoint)
    # TODO for now, model is ignored
    model = None
    train_interactions = torch.load(args.train_interactions)
    #  train_loader = DataLoader(train_data, batch_size=opts.batch_size)
    #  valid_loader = DataLoader(valid_data, batch_size=opts.batch_size)
    return model, train_interactions, data, train_data, test_data


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
    model, train_I, dataset, train_data, test_data = get_model_data(params)
    # check that data is correctly split by verifying that all training data is
    # the same in saved interactions and in the data processed on the fly
    #  n_train_data = train_I.sender_input.size(0)
    #  assert(n_train_data == len(train_data))
    #  for i in range(n_train_data):
    #      assert(torch.all(train_I.sender_input[i] ==
    #          torch.tensor(train_data[i][0][1])))
    arg_counts = count_argument_positions(train_I)
    j = 0
    I = train_I
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
