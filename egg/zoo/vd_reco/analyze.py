# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os
from collections import Counter, defaultdict
import pylev

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.distributions import Categorical
from functools import partial
import numpy as np
from scipy.stats import spearmanr

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
    parser.add_argument('--topsim', action='store_true')
    #  parser.add_argument('train_interactions', type=str)
    args = parser.parse_args(params)
    dirname = os.path.dirname(args.checkpoint)
    dataset_json = os.path.join(dirname, 'data.json')
    with open(dataset_json, 'r') as f:
        json_data = json.load(f)
        dataset_name = json_data["dataset"]
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    hp_json = os.path.join(dirname, 'hp.json')
    if dataset_name == 'proto':
        data_cfg = DataProto.Settings.load(dataset_json)
        dataset, train_data, valid_data, test_data = init_data_proto(data_cfg, 0, 128)
        model = load_game(Hyperparameters.load(hp_json), loss_objs)
        model.load_state_dict(checkpoint.model_state_dict)
    else:
        raise ValueError('Unknown dataset', dataset_name)
    #  train_interactions = torch.load(args.train_interactions)
    return args, dirname, model, dataset, train_data, valid_data, test_data


def count_argument_positions(interactions):
    arguments_counts = Counter()
    position_counts_per_arg = defaultdict(Counter)
    for i, m in enumerate(interactions.message.argmax(2)):
        to_send = interactions.aux["sender_input_to_send"][i]
        #  print(interactions.aux["gram_funcs"][i], i)
        for j, v in enumerate(to_send[1:]):
            if v == 1 or (interactions.aux["gram_funcs"][i][j] >= 0):
            #  if v.item() == 1:
                arg = tuple(interactions.sender_input[i][j].tolist())
                arguments_counts[arg] += 1
                position_counts_per_arg[arg][j] += 1
    return arguments_counts, position_counts_per_arg


def main(params):
    args, dirname, model, dataset, train_data, valid_data, test_data = get_model_data(params)
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

    #  evaluator = Trainer(model, None, train_data, valid_data, 'cpu', None, None, False)
    evaluator = Trainer(model, None, train_data, train_data, 'cpu', None, None, False)
    loss_valid, I = evaluator.eval()
    score_path_json = os.path.join(dirname, 'best_scores.json')
    if os.path.exists(score_path_json):
        with open(score_path_json, 'r') as f:
            best_score = json.load(f)['best_score']
        print("Best score vs eval", loss_valid, best_score)
    else:
        print("Score path not found. Loss:", loss_valid)

    if args.topsim:
        N = len(I.aux)
        pairs = []
        for k in range(100000):
            i, j = np.random.choice(N, 2)
            m_i = I.message[i].argmax(1)
            m_j = I.message[j].argmax(1)
            # the objects will be the objects to send only
            def object_repr(idx):
                to_send_i = I.aux["sender_input_to_send"][idx][1:] > 0
                return (I.sender_input[idx] * to_send_i.unsqueeze(1)).view(-1)
            x_i = object_repr(i) 
            x_j = object_repr(j)
            d_m = pylev.levenshtein(m_i.tolist(), m_j.tolist())
            d_x = (x_i != x_j).int().sum().item()
            pairs.append((d_x, d_m))
        import pdb; pdb.set_trace()
        pairs = np.asarray(pairs)
        print("spearman", spearmanr(pairs[:, 0], pairs[:, 1]))
        exit()

    arg_counts, pos_counts = count_argument_positions(I)
    j = 0
    marker = 99
    try:
        roleset_permutations = dataset.roleset_permutations
    except AttributeError:
        roleset_permutations = None
    idx_arg = {arg: i for i, arg in enumerate([k for k, v in
               arg_counts.most_common()])}

    per_rolesets = defaultdict(list)
    per_to_send = defaultdict(list)
    for i, loss_objs in enumerate(I.aux['loss_objs']):
        loss_objs = loss_objs.item()
        if loss_objs > 16:
            continue
        loss_objs_D = np.around(I.aux["loss_objs_D"][i].numpy(), decimals=3)
        to_send = I.aux["sender_input_to_send"][i][1:].int()
        for k, v in enumerate(to_send):
            if v == 1 or (I.aux["gram_funcs"][i][k]):
                arg = tuple(I.sender_input[i][k].tolist())
                to_send[k] = idx_arg[arg]
            else:
                to_send[k] = -1.
        roleset = int(I.aux['roleset'][i].item())
        # for debugging:
        #  if roleset == 880 or roleset == '880':
        #      print(I.sender_input[i])
        m = I.message[i].argmax(1).tolist()
        if roleset_permutations:
            perm_to_send = tuple(to_send[roleset_permutations[roleset]].tolist())
        else:
            perm_to_send = None
        to_send = tuple(to_send.tolist())
        per_rolesets[roleset].append(
            (m, to_send, perm_to_send, loss_objs, loss_objs_D))
        per_to_send[to_send].append(
            (m, roleset, perm_to_send, loss_objs, loss_objs_D))

    # ORDER BY ROLESET:
    #  for argument, count in arg_counts.most_common(30):
    #      print("Arg #{}: {} ({}) {}".format(j, argument, count,
    #          pos_counts[argument]))
    #      j += 1
        #  for roleset, data in per_rolesets.items():
        #      if len(data) < 2:
        #          continue
        #      sorted_data = sorted(data, key=lambda e: e[1])
        #      print("Roleset", roleset)
        #      for m, to_send, P_to_send, loss, loss_D in sorted_data:
        #          if P_to_send:
        #              print(m, to_send, P_to_send, loss_D, loss)
        #          else:
        #              print(m, to_send, loss_D, loss)
    for argument, count in arg_counts.most_common(30):
        print("Arg #{}: {} ({}) {}".format(j, argument, count,
                  pos_counts[argument]))
        j += 1
        i = idx_arg[argument]
        arg_filtered = [(k, v) for k, v in per_to_send.items() if i in k]
        arg_sorted = sorted(arg_filtered, key=lambda kv: (list(kv[0]).index(i), kv[0]))
        for k, data in arg_sorted:
            to_send = k
            if len(data) < 2:
                continue
            sorted_data = sorted(data, key=lambda e: e[1])
            # sort by roleset
            for m, roleset, _, loss, loss_D in sorted_data:
                print("S={} - m={} - loss_D={} - role={}".format(
                    to_send, m, loss_D, roleset))

    print("END")



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
