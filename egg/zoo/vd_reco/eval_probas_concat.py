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
from torch.utils.data.dataloader import default_collate
from torch.distributions import Categorical
from functools import partial
import numpy as np

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
from .utils_transitivity import compute_transitivity_exact as compute_transitivity
import dit


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
    parser.add_argument('split', type=str)
    parser.add_argument('--truncate', action='store_true',
            help='When concatenation m1.m2 is longer than max_len, truncate. Use training probability distribution on finite-lengths messages.')
    args = parser.parse_args(params)
    np.random.seed(0)

    dirname, hp, model, dataset, train_data, valid_data, test_data = \
        load_model_data_from_cp(args.checkpoint)
    init_common_opts_reloading()

    if args.split == 'train':
        data = train_data
    elif args.split == 'valid':
        data = valid_data
    elif args.split == 'test':
        data = test_data
    else:
        raise ValueError(f"Split {args.split} unknown!")
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
    # check options are correct
    N = len(I.aux['length'])
    # build a dict indexed by speaker's input matrix, which value is another
    # dict indexed by speaker's to send matrix, which value is the message sent
    # by the speaker.
    # this way we can find all the messages associated with a given input
    # matrix, when the different objects to send vary.
    gb_sender_input = defaultdict(dict)
    ids_to_inputs = defaultdict(dict)
    symbol_role_mat = np.zeros((3, hp.vocab_size))
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
            # Compute MI between theta role and individual symbols
            #  if sum(to_send) == 1:
            #      role = (to_send == 1).nonzero(as_tuple=False)[0]
            #      # WARNING: the following counts one symbol per message.
            #      symbol_role_mat[role, msg] += 1
    # MI role/symbol
    #  symbol_role_mat = symbol_role_mat / symbol_role_mat.sum()
    #  distrib = dit.Distribution.from_ndarray(symbol_role_mat)
    #  MI_symbol_role = dit.shannon.mutual_information(distrib, [0], [1])
    #  entropy_role = dit.shannon.entropy(distrib, [0])

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
                m_ref = v.get(to_send_sum, None)

            if m_ref is not None:
                # get its id:
                ID = ids_to_inputs[sender_input][to_send_sum]
                m1 = remove_padding_list(msg_1.tolist())
                m2 = remove_padding_list(msg_2.tolist())
                def to_array(l):
                    if args.truncate:
                        return torch.tensor(pad_list(l, hp.max_len + 1))
                    else:
                        return torch.tensor(pad_list(l, 2*(hp.max_len) + 1))
                m12, m21 = to_array(m1 + m2), to_array(m2 + m1)
                if args.truncate:
                    m12 = m12[:hp.max_len + 1]
                    m21 = m21[:hp.max_len + 1]
                    m12[-1] = 0
                    m21[-1] = 0
                m_ref = remove_padding_list(m_ref.tolist())
                all_msgs = (
                    to_array(m_ref),  # the reference message
                    m12,
                    m21,
                    # original single messages
                    to_array(m1),  
                    to_array(m2),
                )
                triplets_messages.append((
                    ID,
                    all_msgs,  
                    np.asarray([n_send_1, n_send_2]),  # indicating roles
                    (len(m1), len(m2)),
                ))

    data = []
    for ID, msgs, to_send, lengths in triplets_messages:
        sender_in = dataset[ID][0]
        labels = dataset[ID][1]
        receiver_in = dataset[ID][2]
        K = len(msgs) 
        for msg in msgs:
            data.append((sender_in, labels, receiver_in, msg, to_send, ID,
                lengths))

    data = DataLoader(data, shuffle=False, batch_size=K*128)  #, collate_fn=collate)
    def create_storage_analysis():
        return [defaultdict(list) for _ in range(6)] + [Counter() for _ in range(2)]
    diffs_simple_concatenation = create_storage_analysis()
    pair_roles_counts = Counter()  
    N_dbg_I = 1
    dbg_I = I.aux['ids'][:N_dbg_I]
    for i in range(N_dbg_I):
        ID = I.aux['ids'][i]
        print("I:", ID, [k + "=" + str(v[i]) for k, v in I.aux.items()])

    def find_loss(ID):
        found = (I.aux['ids'] == ID).nonzero(as_tuple=False).item()
        return found, I.aux['loss'][found]


    with torch.no_grad():
        for sender_in, labels, receiver_in, msgs, to_send, ID, lengths in data:
            S = to_send[0::K]  # TODO assert all close with [1::K], [2::K]
            probas = model.eval_proba_sender(sender_in, msgs, has_max_len=False)
            losses, sum_loss, loss = model.eval_loss_receiver(sender_in,
                    labels, receiver_in, msgs, ID)
            ref_proba, p12, p21 = probas[0::K], probas[1::K], probas[2::K]
            p1, p2 = probas[3::K], probas[4::K]
            ref_loss, l12, l21 = loss[0::K], loss[1::K], loss[2::K]
            l1, l2 = loss[3::K], loss[4::K]

            def compute_store_metrics(p12, p21, l12, l21,
                    l_diffs, lP_diffs, l_best_diffs, lP_best_diffs,
                    l_sm_diffs, lP_sm_diffs, 
                    ordering_loss_diffs, ordering_prob_diffs,
                ):
                """ Compute various metrics and store them.
                """
                # p12_diff = log(p(m12)/p(m*)), if >0 then p(m12) > p(m*) where m*
                # is the reference message, the one actually output by the speaker
                p12_diff = p12 - ref_proba
                p21_diff = p21 - ref_proba
                # l12_diff > 0 ↔ l12 > ref_loss
                l12_diff = ref_loss - l12
                l21_diff = ref_loss - l21
                # removed
                #  l12_diff_norm = l12_diff / ref_loss
                #  l21_diff_norm = l21_diff / ref_loss
                for i in range(S.size(0)):
                    _, ref_l_sanity_check = find_loss(ID[i*K])
                    assert(torch.allclose(ref_l_sanity_check, sum_loss[i*K]))
                    pos1, pos2 = S[i].tolist()
                    pos12, pos21 = (pos1, pos2), (pos2, pos1)
                    idx_sm_diff = tuple(sorted(pos12))
                    lP_diffs[pos12].append(p12_diff[i].item())
                    lP_diffs[pos21].append(p21_diff[i].item())
                    l_diffs[pos12].append(l12_diff[i].item())
                    l_diffs[pos21].append(l21_diff[i].item())
                    # these are concatenability metrics, since they min/max over order
                    l_best_diffs[idx_sm_diff].append(
                        ref_loss[i].item() - min(l12[i].item(), l21[i].item())
                    )
                    lP_best_diffs[idx_sm_diff].append(
                        max(p12[i].item(), p21[i].item()) - ref_proba[i].item()
                    )
                    # we can also check how the concatenation compares to using a
                    # single message.
                    # l_sm_diffs > 0 ↔ min(l1, l2) > ref_loss
                    # so real message is better than either m1 or m2 when > 0
                    l_sm_diffs[idx_sm_diff].append(
                        min(l1[i].item(), l2[i].item()) - ref_loss[i].item()
                    )
                    lP_sm_diffs[idx_sm_diff].append(
                        ref_proba[i].item() - max(p1[i].item(), p2[i].item()) 
                    )
                    if l12[i].item() < l21[i].item():
                        ordering_loss_diffs[pos12] += 1
                    else:
                        ordering_loss_diffs[pos21] += 1
                    if p12[i].item() > p21[i].item():
                        ordering_prob_diffs[pos12] += 1
                    else:
                        ordering_prob_diffs[pos21] += 1

            compute_store_metrics(
                p12, p21, l12, l21, 
                *diffs_simple_concatenation,
            )
            for i in range(S.size(0)):
                #  if ID[::K][i] in dbg_I:
                #      print(f"ID={ID[::K][i]}:", dbg_I)
                #      print([k + "=" + str(v[::K][i]) for k, v in losses.items() if
                #          type(v) != int])
                pos1, pos2 = S[i].tolist()
                pos12, pos21 = (pos1, pos2), (pos2, pos1)
                idx_sm_diff = tuple(sorted(pos12))
                str_idx_sm_diff = str(idx_sm_diff[0]) + "," + str(idx_sm_diff[1])
                pair_roles_counts[str_idx_sm_diff] += 1

    filename_wo_ext = 'concat_' + args.split
    if args.truncate:
        filename_wo_ext += '_trunc'
    compo_json = os.path.join(dirname, filename_wo_ext + '.json')

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

    #  sum_unnorm_dists = sum_global_min(dists)
    #  sum_norm_dists = sum_global_min(normalized_dists)
    #  sum_local_unnorm_dists = sum_local_min(dists)
    #  sum_local_norm_dists = sum_local_min(normalized_dists)
    #  n_combinations = [len(dists[e]) for e in [(0,1), (0,2), (1,2)]]
    #  n = sum(n_combinations)
    def compute_T(diffs):
        n_compared = sum(diffs.values())
        diffs = {','.join([str(e) for e in k]): v for k, v in
                  diffs.items()}
        transitivity, edges = compute_transitivity(diffs, total=n_compared)
        edges = [str(a) + "," + str(b) for a, b in edges]
        return {'transitivity': transitivity,
                'n_compared': n_compared,
                'edges': edges,
        }

    def prepare_dict(d):
        prepared_d = {}
        for k, v in d.items():
            key = ','.join([str(e) for e in k])
            prepared_d[key + '_mean'] = torch.tensor(v).mean().item()
            prepared_d[key + '_std'] = torch.tensor(v).std().item()
            prepared_d[key + '_n'] = torch.tensor(v).size(0)
        return prepared_d

    def add_to_data(list_containers, data, suffix):
        data['loss_diffs' + suffix] = prepare_dict(list_containers[0])
        data['log_prob_diffs' + suffix] = prepare_dict(list_containers[1])
        data['loss_best_diffs' + suffix] = prepare_dict(list_containers[2])
        data['log_prob_best_diffs' + suffix] = prepare_dict(list_containers[3])
        data['loss_sm_diffs' + suffix] = prepare_dict(list_containers[4])
        data['log_prob_sm_diffs' + suffix] = prepare_dict(list_containers[5])
        data['transitivity_listener' + suffix] = compute_T(list_containers[6])
        data['transitivity_speaker' + suffix] = compute_T(list_containers[7])

    with open(compo_json, 'w') as fp:
        d = {}
        add_to_data(diffs_simple_concatenation, d, suffix='')
        d['pair_roles_counts'] = dict(pair_roles_counts)
        print(d)
        json.dump(d, fp)
    exit()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
