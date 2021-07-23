# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os
from collections import Counter, defaultdict
import pylev
from .levenshtein import wfi_levenshtein
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
from .archs_protoroles import Hyperparameters, load_game
from .train_vl import loss_objs
#  from .data_proto import init_data as init_data_proto
from .data_proto import Data as DataProto, init_data as init_data_proto
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from simple_parsing import ArgumentParser
from egg.core import Trainer
import egg.core.util as util
from egg.core.distributed import not_distributed_context
from .callbacks import entropy_list


def normalized_levenshtein(msg1, msg2, substitution_cost):
    d = wfi_levenshtein(msg1, msg2, substitution_cost)
    denom = len(msg1) + len(msg2)
    if denom == 0:
        return 0
    return d / float(denom)


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

def get_model_data(params):
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--topsim', action='store_true')
    parser.add_argument('--topsim_option', default='')
    parser.add_argument('--compo', action='store_true')
    parser.add_argument('--compo_option', default='')
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
    np.random.seed(0)
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
    split = 'train'
    if split == 'train':
        data = train_data
    elif split == 'valid':
        data = valid_data
    elif split == 'test':
        data = test_data
    evaluator = Trainer(model, None, train_data, data, 'cpu', None, None, False)
    loss_valid, I = evaluator.eval()
    score_path_json = os.path.join(dirname, 'best_scores.json')
    if os.path.exists(score_path_json):
        with open(score_path_json, 'r') as f:
            best_score = json.load(f)['best_score']
        print("Best score vs eval", loss_valid, best_score)
    else:
        print("Score path not found. Loss:", loss_valid)
    if args.topsim_option:
        assert(args.topsim)
    if args.compo_option:
        assert(args.compo)

    n = 0
    n_ctx = 0
    options = args.topsim_option.split(',')
    if args.topsim:
        if 'sub2' in options:
            sub_cost = 2
        else:
            sub_cost = 1
        N = len(I.aux['length'])
        pairs = []
        pairs_ctx = []
        if 'norm' in options:
            dist_f = normalized_levenshtein
        else:
            dist_f = wfi_levenshtein
        for k in range(100000):
            i, j = np.random.choice(N, 2)
            m_i = I.message[i].argmax(1)
            m_j = I.message[j].argmax(1)
            # the objects will be the objects to send only
            def object_repr(idx):
                if 'no_filter' in options:
                    return I.sender_input[idx]
                elif '1on1':
                    to_send = I.aux["sender_input_to_send"][idx][1:] > 0
                    idx_to_send = to_send.tolist().index(1)
                    return I.sender_input[idx][idx_to_send]
                else:
                    to_send = I.aux["sender_input_to_send"][idx][1:] > 0
                    return (I.sender_input[idx] * to_send.unsqueeze(1)).view(-1)
            def context_repr(idx):
                to_send = I.aux["sender_input_to_send"][idx][1:] > 0
                # what is NOT to send is the context
                context = (~ to_send).unsqueeze(1)
                flat_context = (I.sender_input[idx] * context).view(-1)
                return flat_context

            to_send_i = I.aux["sender_input_to_send"][i][1:]
            to_send_j = I.aux["sender_input_to_send"][j][1:]
            d_m = dist_f(remove_padding_list(m_i.tolist()),
                         remove_padding_list(m_j.tolist()),
                         substitution_cost=sub_cost)

            if (to_send_i == to_send_j).all():
                d_x_ctx = (context_repr(i) != context_repr(j)).int().sum().item()
                pairs_ctx.append((d_x_ctx, d_m))
                n_ctx += 1
            if '1on1' in options:
                n_to_send_i = (to_send_i > 0).sum()
                n_to_send_j = (to_send_j > 0).sum()
                if n_to_send_i != 1 or n_to_send_j != 1:
                    continue
            n+=1
            x_i = object_repr(i) 
            x_j = object_repr(j)
            d_x = (x_i != x_j).int().sum().item()
            pairs.append((d_x, d_m))
        pairs = np.asarray(pairs)
        pairs_ctx = np.asarray(pairs_ctx)
        spearman = spearmanr(pairs[:, 0], pairs[:, 1])
        spearman_ctx = spearmanr(pairs_ctx[:, 0], pairs_ctx[:, 1])
        if options:
            topsim_json = os.path.join(
                dirname, 'topsim_' + args.topsim_option + '.json',
            )
        else:
            topsim_json = os.path.join(dirname, 'topsim.json')
        with open(topsim_json, 'w') as fp:
            data = {
                'split': split,
                'spearman': spearman.correlation, 'pvalue': spearman.pvalue, 'n': n,
                'spearman_ctx': spearman_ctx.correlation,
                'pvalue_ctx': spearman_ctx.pvalue, 'n_ctx': n_ctx,
            }
            json.dump(data, fp)
        print("Correlation to send={}, correlation context={}".format(
              spearman, spearman_ctx))
        exit()
    elif args.compo:
        options = args.compo_option.split(',')
        # check options are correct
        sub_cost = 2
        N = len(I.aux['length'])
        # we're going to group examples by sender_input
        gb_sender_input = defaultdict(dict)
        for i in range(N):
            to_send = I.aux["sender_input_to_send"][i][1:]
            sender_input = tuple(I.sender_input[i].view(-1).tolist())
            msg = I.message[i].argmax(1)
            gb_sender_input[sender_input][tuple(to_send.tolist())] = msg
        dists = defaultdict(list)
        normalized_dists = defaultdict(list)
        for sender_input, v in gb_sender_input.items():
            #  v = sorted(v, key=lambda e: e[0].sum())
            v_one_object = [e for e in v.items() if sum(e[0]) == 1]
            #  roles = [list(e[0]).index(1) for e in v_one_object]
            #  # there are duplicates, simply b/c there are identical annotations
            #  combinations = itertools.combinations(roles, 2)
            combinations = itertools.combinations(range(len(v_one_object)), 2)
            for combination in combinations:
                i, j = combination
                if len(v_one_object) >= 2:
                    to_send_1 = v_one_object[i][0]
                    msg_1 = v_one_object[i][1]
                    to_send_2 = v_one_object[j][0]
                    msg_2 = v_one_object[j][1]
                    n_send_1 = list(to_send_1).index(1)
                    n_send_2 = list(to_send_2).index(1)
                    to_send_sum = np.asarray(to_send_1) + np.asarray(to_send_2)
                    found = False
                    for to_send, msg in v.items():
                        if np.allclose(to_send_sum, to_send):
                            msg_sum = msg
                            found = True
                            break
                if found:
                    n = len(msg_1.tolist())
                    cut_msg_1 = remove_padding_list(msg_1.tolist())
                    cut_msg_2 = remove_padding_list(msg_2.tolist())
                    concats = [cut_msg_1 + cut_msg_2,
                               cut_msg_2 + cut_msg_1]
                    order = [(n_send_1, n_send_2), (n_send_2, n_send_1)]
                    msg_sum = remove_padding_list(msg_sum)
                    for o, concatenation in zip(order, concats):
                        n_dist = normalized_levenshtein(msg_sum, concatenation, 2)
                        dist = wfi_levenshtein(msg_sum, concatenation,
                                               substitution_cost=2)
                        dists[o].append(dist)
                        normalized_dists[o].append(n_dist)
                    #  i_min = np.argmin(D)
                    #  i_max = np.argmax(D)
                    #  if D[i_min] != D[i_max]:
                    #      concat_ordering[order[i_min]] += 1
                    #  smallest_D = D[i_min]
        if options:
            compo_json = os.path.join(
                dirname, 'compo_' + args.compo_option + '.json',
            )
        else:
            compo_json = os.path.join(dirname, 'compo.json')

        # For each set of pairs (like {0,1}), pick the order that minimizes the
        # distance (like (0,1)). Sum those distances.
        def sum_global_min(A):
            return int(
                min(np.sum(A[(0,1)]), np.sum(A[(1,0)])) + 
                min(np.sum(A[(0,2)]), np.sum(A[(2,0)])) + 
                min(np.sum(A[(1,2)]), np.sum(A[(2,1)]))
            )
        sum_unnorm_dists = sum_global_min(dists)
        sum_norm_dists = sum_global_min(normalized_dists)
        n_combinations = [len(dists[e]) for e in [(0,1), (0,2), (1,2)]]
        n = sum(n_combinations)
        #  concat_ordering = {','.join([str(e) for e in k]): v for k, v in
        #          concat_ordering.items()}
        with open(compo_json, 'w') as fp:
            data = {
                'n': n,
                'sum_unnorm': sum_unnorm_dists / float(n),
                'sum_norm': sum_norm_dists / float(n),
                #  'n': len(distances),
                #  'mean': distances.mean(),
                #  'std': distances.std(),
                #  'order': concat_ordering,

            }
            print(data)
            json.dump(data, fp)

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
    single_object_messages = defaultdict(Counter)
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
            #      import pdb; pdb.set_trace()
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
    for argument, count in arg_counts.most_common():
        all_msg = single_object_messages[j]
        n = sum(all_msg.values())
        H = entropy_list(all_msg.values())
        H_per_msg.append((argument, n, H))
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

    H_json = os.path.join(dirname, 'per_arg_H_msg.json')
    with open(H_json, 'w') as fp:
        sum_n = sum([n for _, n, _ in H_per_msg])
        weighted_sum_H = sum([H*n for _, n, H in H_per_msg])
        avg_H_freq_weighted = float(weighted_sum_H) / float(sum_n)
        data = {'H_per_msg': H_per_msg, 'freq_weighted_avg_H':
                avg_H_freq_weighted}
        json.dump(data, fp)
    print("END")



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])