# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os
from collections import Counter, defaultdict
import itertools

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
from .train_vl import loss_ordered, loss_unordered
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from simple_parsing import ArgumentParser
from egg.core import Trainer
from .callbacks import entropy_list
from .utils import load_model_data_from_cp, init_common_opts_reloading


def remove_padding_list(msg):
    """ Return same list without the 0s.
    """
    c = 0
    for m in reversed(msg):
        if m != 0:
            break
        c += 1
    return msg[:-c]

def pad_list(msg, n):
    """ Pad to n with 0s.
    """
    if len(msg) == n:
        return msg
    return msg + list((0,) * (n - len(msg)))

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
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--beam-search', default=0, type=int)
    args = parser.parse_args(params)
    np.random.seed(0)

    dirname, hp, model, dataset, train_data, valid_data, test_data = \
        load_model_data_from_cp(args.checkpoint)
    init_common_opts_reloading()

    #  evaluator = Trainer(model, None, train_data, valid_data, 'cpu', None, None, False)
    split = 'train'
    if split == 'train':
        data = train_data
    elif split == 'valid':
        data = valid_data
    elif split == 'test':
        data = test_data

    if args.beam_search > 0:
        model.sender.generate_style = 'beam_search'
        model.sender.beam_size = args.beam_search

    evaluator = Trainer(model, None, train_data, data, 'cpu', None, None, False)
    loss_valid, I = evaluator.eval()
    score_path_json = os.path.join(dirname, 'best_scores.json')
    if os.path.exists(score_path_json):
        with open(score_path_json, 'r') as f:
            best_score = json.load(f)['best_score']
        print("Best score vs eval", loss_valid, best_score)
    else:
        print("Score path not found. Loss:", loss_valid)

    n = 0
    n_ctx = 0

    arg_counts, pos_counts = count_argument_positions(I)
    marker = 99
    try:
        roleset_permutations = dataset.roleset_permutations
    except AttributeError:
        roleset_permutations = None
    idx_arg = {arg: i for i, arg in enumerate([k for k, v in
               arg_counts.most_common()])}

    per_rolesets = defaultdict(list)
    per_to_send = defaultdict(list)
    single_object_messages = defaultdict(Counter)
    single_object_roles = defaultdict(Counter)
    for i, loss_objs in enumerate(I.aux['loss_objs']):
        loss_objs = loss_objs.item()
        #  if loss_objs > 16:
        #      continue
        loss_objs_D = np.around(I.aux["loss_objs_D"][i].numpy(), decimals=3)
        to_send = I.aux["sender_input_to_send"][i][1:].int()
        to_send_orig = to_send.tolist()
        for k, v in enumerate(to_send):
            if I.aux["gram_funcs"][i][k] >= 0:
                arg = tuple(I.sender_input[i][k].tolist())
                to_send[k] = idx_arg[arg]
            else:
                to_send[k] = -1.
            #  if to_send[k] == 4:
        roleset = int(I.aux['roleset'][i].item())
        # for debugging:
        #  if roleset == 880 or roleset == '880':
        #      print(I.sender_input[i])
        m = I.message[i].argmax(1).tolist()
        if roleset_permutations:
            perm_to_send = tuple(to_send[roleset_permutations[roleset]].tolist())
        else:
            perm_to_send = to_send_orig
            # bad name. This code is messy b/c of permutations...
        if sum(to_send_orig) == 1:
            idx_to_send = to_send_orig.index(1)
            arg_i = to_send[idx_to_send].item()
            single_object_messages[arg_i][tuple(m)] += 1
            single_object_roles[arg_i][idx_to_send] += 1
            
        to_send_t = tuple(to_send.tolist())
        per_rolesets[roleset].append(
            (m, to_send_t, perm_to_send, loss_objs, loss_objs_D))
        per_to_send[to_send_t].append(
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
    print_limit = 30
    H_per_msg = []
    j = 0
    for argument, count in arg_counts.most_common():
        all_msg = single_object_messages[j]
        all_roles = single_object_roles[j]
        # number of datapoints where 1 object must be sent, and this object is
        # the j-th most frequent object. (count: object is present but it might
        # not be to send, or there might be several objects to send, etc.)
        n = sum(all_msg.values())
        H = entropy_list(all_msg.values())
        H_roles = entropy_list(all_roles.values())
        # interesting to compare H_1, the conditional entropy of the message given a
        # single input to the conditional entropy of the role given a single
        # input. If synthetic, H_1 should not go below this second metric,
        # because each message should at least encode the role.
        H_per_msg.append((argument, n, H, H_roles))
        if j <= print_limit:  
            print("Arg #{}: {} ({}) {}".format(j, argument, count,
                      pos_counts[argument]))
            print(all_msg)
            print("Obj {}: #n={}, H={}".format(argument, n, H))
        j += 1
        i = idx_arg[argument]
        arg_filtered = [(k, v) for k, v in per_to_send.items() if i in k]
        arg_sorted = sorted(arg_filtered, key=lambda kv: (list(kv[0]).index(i), kv[0]))
        for k, data in arg_sorted:
            objs = k
            #  if len(data) < 2:
            #      continue
            sorted_data = sorted(data, key=lambda e: e[1])
            # sort by roleset
            if j <= print_limit:
                for m, roleset, ts, loss, loss_D in sorted_data:
                    print("S={} {} - m={} - loss_D={} - role={}".format(
                        objs, ts, m, loss_D, roleset))
    print("END")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
