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
from torch.utils.data.dataloader import default_collate
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
from .utils_transitivity import compute_transitivity


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
    parser.add_argument('--compo_option', default='')
    args = parser.parse_args(params)
    np.random.seed(0)

    dirname, hp, model, dataset, train_data, valid_data, test_data = \
        load_model_data_from_cp(args.checkpoint)
    max_len = hp.max_len
    init_common_opts_reloading()

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
    # I save the ids as floats (because EGG wants to compute means of all the
    # things saved in aux), but actually they should be intepreted as ints:
    # they're IDs of datapoints in the the dataset
    I.aux['ids'] = I.aux['ids'].int()
    score_path_json = os.path.join(dirname, 'best_scores.json')
    if os.path.exists(score_path_json):
        with open(score_path_json, 'r') as f:
            best_score = json.load(f)['best_score']
        print("Best score vs eval", loss_valid, best_score)
    else:
        print("Score path not found. Loss:", loss_valid)

    n = 0
    n_ctx = 0
    options = args.compo_option.split(',')
    # check options are correct
    sub_cost = 2
    N = len(I.aux['length'])
    # build a dict indexed by speaker's input matrix, which value is another
    # dict indexed by speaker's to send matrix, which value is the message sent
    # by the speaker.
    # this way we can find all the messages associated with a given input
    # matrix, when the different objects to send vary.
    gb_sender_input = defaultdict(dict)
    ids_to_inputs = defaultdict(dict)
    with torch.no_grad():
        for i in range(N):
            to_send = I.aux["sender_input_to_send"][i][1:]
            sender_input = tuple(I.sender_input[i].view(-1).tolist())
            msg = I.message[i].argmax(1)
            to_send_tuple = tuple(to_send.tolist())
            already_ID = ids_to_inputs[sender_input].get(to_send_tuple, None) 
            if already_ID is not None:
                continue
               #  print("ALREADY PRESENT", already_ID)
            else:
               gb_sender_input[sender_input][to_send_tuple] = msg
               ids_to_inputs[sender_input][to_send_tuple] = I.aux['ids'][i].item()
            # TODO problem: what if tuple(...) already exists in 
            # the dictionary? it erases it...


    dists = defaultdict(list)
    normalized_dists = defaultdict(list)
    concat_ordering = Counter()
    triplets_messages = []  # tuple of (ID, M):
    # where M is a tuple containing
    # the regular message, concat_order_1 and concat order 2 message. 
    for sender_input, v in gb_sender_input.items():
        # for a given input, gather the to send matrices such that 1 object has
        # to be sent.
        v_one_object = [e for e in v.items() if sum(e[0]) == 1]
        #  roles = [to_send.index(1) for to_send, _ in v_one_object]
        # compute all combinations of 2 objects to be sent, and try to find
        # the corresponding message
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
                to_send_sum = tuple(np.asarray(to_send_1) + np.asarray(to_send_2))
                msg_sum = v.get(to_send_sum, None)

            if msg_sum is not None:
                # get its id:
                ID = ids_to_inputs[sender_input][to_send_sum]
                n = len(msg_1.tolist())
                cut_msg_1 = remove_padding_list(msg_1.tolist())
                cut_msg_2 = remove_padding_list(msg_2.tolist())
                concats = [cut_msg_1 + cut_msg_2,
                           cut_msg_2 + cut_msg_1]
                order = [(n_send_1, n_send_2), (n_send_2, n_send_1)]
                msg_sum = remove_padding_list(msg_sum.tolist())
                D = []
                for o, concatenation in zip(order, concats):
                    n_dist = normalized_levenshtein(msg_sum, concatenation, 2)
                    dist = wfi_levenshtein(msg_sum, concatenation, 2)
                    D.append(dist)
                    dists[o].append(dist)
                    normalized_dists[o].append(n_dist)
                i_min = np.argmin(D)
                i_max = np.argmax(D)
                if D[i_min] != D[i_max]:
                   concat_ordering[order[i_min]] += 1
                def to_array(l):
                    return torch.tensor(pad_list(l, 2*(hp.max_len) + 1))
                all_msgs = (to_array(msg_sum),  # the reference message
                    to_array(concats[0]),  # should be [msg1,msg2] concat
                    to_array(concats[1]),  # should be [msg1,msg2] concat
                    to_array(cut_msg_1),  # original msg1, but need to repad
                    to_array(cut_msg_2),
                )
                triplets_messages.append((
                    ID,
                    all_msgs,  
                    np.asarray([n_send_1, n_send_2]),  # indicating roles
                ))

    #  # use DataLoader:
    #  def collate(d):
    #      import pdb; pdb.set_trace()

    data = []
    for ID, msgs, to_send in triplets_messages:
        sender_in = dataset[ID][0]
        labels = dataset[ID][1]
        receiver_in = dataset[ID][2]
        K = len(msgs) 
        for msg in msgs:
            data.append((sender_in, labels, receiver_in, msg, to_send, ID))

    data = DataLoader(data, shuffle=False, batch_size=K*128)  #, collate_fn=collate)
    log_prob_diffs = defaultdict(list)  # compare against reference message
    log_prob_sm_diffs = defaultdict(list)  # against best single message
    loss_diffs = defaultdict(list)
    loss_sm_diffs = defaultdict(list)
    N_dbg_I = 1
    dbg_I = I.aux['ids'][:N_dbg_I]
    for i in range(N_dbg_I):
        ID = I.aux['ids'][i]
        print("I:", ID, [k + "=" + str(v[i]) for k, v in I.aux.items()])

    def find_loss(ID):
        found = (I.aux['ids'] == ID).nonzero(as_tuple=False).item()
        return found, I.aux['loss'][found]


    with torch.no_grad():
        for sender_in, labels, receiver_in, msgs, to_send, ID in data:
            S = to_send[0::K]  # TODO assert all close with [1::K], [2::K]

            probas = model.eval_proba_sender(sender_in, msgs)
            losses, sum_loss, loss = model.eval_loss_receiver(sender_in,
                    labels, receiver_in, msgs, ID)
            ref_proba, p12, p21 = probas[0::K], probas[1::K], probas[2::K]
            p1, p2 = probas[3::K], probas[4::K]
            ref_loss, l12, l21 = loss[0::K], loss[1::K], loss[2::K]
            l1, l2 = loss[3::K], loss[4::K]
            for i in range(S.size(0)):
                if ID[::K][i] in dbg_I:
                    print(f"ID={ID[::K][i]}:", dbg_I)
                    print([k + "=" + str(v[::K][i]) for k, v in losses.items() if
                        type(v) != int])
                n_send_1 = S[i, 0].item()
                n_send_2 = S[i, 1].item()
                p12_diff = (p12[i] - ref_proba[i]).item()  
                # p12_diff = log(p(m12)/p(m*)), if >0 then p(m12) > p(m*) where m*
                # is the reference message, the one actually output by the speaker
                p21_diff = (p21[i] - ref_proba[i]).item()
                log_prob_diffs[(n_send_1, n_send_2)].append(p12_diff)
                log_prob_diffs[(n_send_2, n_send_1)].append(p21_diff)
                idx_sm_diff = tuple(sorted((n_send_1, n_send_2)))
                log_prob_sm_diffs[idx_sm_diff].append(
                    max(p1[i].item(), p2[i].item()) - ref_proba[i]
                )
                loss_sm_diffs[idx_sm_diff].append(
                    min(l1[i].item(), l2[i].item()) - ref_loss[i]
                )
                _, ref_l_sanity_check = find_loss(ID[i*K])
                assert(torch.allclose(ref_l_sanity_check, sum_loss[i*K]))

                l12_diff = (l12[i] - ref_loss[i]).item()
                # l12_diff > 0 ↔ l12 > ref_loss
                l21_diff = (l21[i] - ref_loss[i]).item()
                loss_diffs[(n_send_1, n_send_2)].append(l12_diff)
                loss_diffs[(n_send_2, n_send_1)].append(l21_diff)
    import pdb; pdb.set_trace()

    if options:
        compo_json = os.path.join(
            dirname, 'compo_' + args.compo_option + '.json',
        )
    else:
        compo_json = os.path.join(dirname, 'compo.json')

    # For each set of pairs (like {0,1}), pick the order that minimizes the
    # distance (like (0,1)). Sum those distances.
    def sum_global_min(A):
        A1 = np.vstack([A[(0, 1)], A[(1, 0)]])
        A2 = np.vstack([A[(0, 2)], A[(2, 0)]])
        A3 = np.vstack([A[(2, 1)], A[(1, 2)]])
        return (A1.sum(1).min() + A2.sum(1).min() + A3.sum(1).min())

    def sum_local_min(A):
        A1 = np.vstack([A[(0, 1)], A[(1, 0)]])
        A2 = np.vstack([A[(0, 2)], A[(2, 0)]])
        A3 = np.vstack([A[(2, 1)], A[(1, 2)]])
        return (A1.min(0).sum() + A2.min(0).sum() + A3.min(0).sum())

    n_compared = sum([len(v) for v in dists.values()])
    sum_unnorm_dists = sum_global_min(dists)
    sum_norm_dists = sum_global_min(normalized_dists)
    sum_local_unnorm_dists = sum_local_min(dists)
    sum_local_norm_dists = sum_local_min(normalized_dists)
    n_combinations = [len(dists[e]) for e in [(0,1), (0,2), (1,2)]]
    n = sum(n_combinations)
    concat_ordering = {','.join([str(e) for e in k]): v for k, v in
              concat_ordering.items()}
    transitivity, edges = compute_transitivity(concat_ordering,
            total=n_compared)
    with open(compo_json, 'w') as fp:
        data = {
            'n': n,
            'sum_unnorm': sum_unnorm_dists / float(n),
            'sum_local_unnorm': sum_local_unnorm_dists / float(n),
            'sum_norm': sum_norm_dists / float(n),
            'sum_local_norm': sum_local_norm_dists / float(n),
            'order': concat_ordering,
            'n_compared': n_compared,
            'transitivity': transitivity,
            #  'n': len(distances),
            #  'mean': distances.mean(),
            #  'std': distances.std(),

        }
        print(data)
        json.dump(data, fp)
    exit()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
