# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os
from collections import Counter, defaultdict, namedtuple
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
from simple_parsing import ArgumentParser
from egg.core import Trainer
from egg.core.interaction import Interaction
from .utils import load_model_data_from_cp, init_common_opts_reloading


def main(params):
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--split', default='train')
    parser.add_argument('--invert-mask', action='store_true',
            help="Invert speaker's mask for debugging")
    args = parser.parse_args(params)
    np.random.seed(0)

    dirname, hp, model, dataset, train_data, valid_data, test_data = \
        load_model_data_from_cp(args.checkpoint)
    max_len = hp.max_len
    init_common_opts_reloading()

    if args.split == 'train':
        data = train_data
    elif args.split == 'valid':
        data = valid_data
    elif args.split == 'test':
        data = test_data
    else:
        raise ValueError(f"Unknown split {args.split}")
    device = torch.device('cuda')
    if args.invert_mask:
        model.sender.agent.set_mask_inverse_debug(True)
    model.sender.set_test_time_sampling(True)
    evaluator = Trainer(model, None, train_data, data, device, None, None, False)
    list_interactions = []
    K_MC = 1
    loss_valid, I = evaluator.eval(n_times=K_MC)
    print("n_messages={I.message.size()}")
    # I save the ids as floats (because EGG wants to compute means of all the
    # things saved in aux), but actually they should be intepreted as ints:
    # they're IDs of datapoints in the the dataset
    I.aux['ids'] = I.aux['ids'].int()
    # Sanity check
    score_path_json = os.path.join(dirname, 'best_scores.json')
    if os.path.exists(score_path_json):
        with open(score_path_json, 'r') as f:
            best_score = json.load(f)['best_score']
        print("Best score vs eval", loss_valid, best_score)
        #  if args.split == 'valid':
            #  assert(abs(float(loss_valid) - float(best_score)) < 1e-5)
    else:
        print("Score path not found. Loss:", loss_valid)

    # compute mapping from ids in the dataloader to probabilities of the
    # messages that are decoded.
    id_dataset2ids_dataloader = defaultdict(list)
    id_dataloader2id_dataset = {}
    batch_size = data.batch_size
    n = len(data.dataset)
    incomplete_batch_size = n % batch_size
    for i, j in enumerate(I.aux['ids']):
        id_dataset2ids_dataloader[int(j)].append(i)
        id_dataloader2id_dataset[i] = int(j)
    inputs_entity_role = defaultdict(set)
    B_log_probas = defaultdict(lambda: defaultdict(list))
    #  B_log_probas = defaultdict(list)
    #  detect_identical = Counter()

    Entry = namedtuple('Entry',
        ['id', 'msg', 'log_p'])

    with torch.no_grad():
        for inputs_S, labels, inputs_R, id_ in data:
            to_send = inputs_S[3][:, 1:]
            for j in range(0, K_MC):
                #  print(f"j={j}")
                ids_dataloader = [id_dataset2ids_dataloader[i.item()][j] for i in id_]
                # sanity check, to verify that the ids are correct:
                rolesets = labels[0]
                for i in ids_dataloader:
                    if i > I.aux['roleset'].size(0):
                        import pdb; pdb.set_trace()
                also_rolesets = torch.stack([I.aux['roleset'][i] for i in
                    ids_dataloader])
                if (rolesets + 1 != also_rolesets).all():
                    import pdb; pdb.set_trace()
                #  assert((rolesets + 1 == also_rolesets).all())
                # select subset of datapoints where only 1 entity should be sent
                single = (to_send.sum(1) == 1)
                ids_subset = torch.tensor(ids_dataloader)[single].tolist()
                #  ids_dataset_tmp_dbg = [id_dataloader2id_dataset[ii] for ii in ids_subset]
                subset_msgs = torch.stack([I.message[i] for i in
                    ids_subset]).argmax(2).to(device)
                subset_inputs_S = tuple([e[single].to(device) for e in inputs_S])
                subset_log_probas = model.eval_proba_sender(subset_inputs_S, subset_msgs)
                subset_roles = to_send[single].nonzero(as_tuple=False)[:, 1]
                # gather all the inputs that share the same entity/role pair to send
                for i, log_p in enumerate(subset_log_probas):
                    role = subset_roles[i].item()
                    entity_vector = tuple(subset_inputs_S[1][i][role].tolist())
                    idx = (role, entity_vector)
                    tuple_msg = tuple(subset_msgs[i].tolist())
                    roleset = subset_inputs_S[0][i].item()
                    id_dataset = id_dataloader2id_dataset[ids_subset[i]]
                    #  full_mat = tuple(subset_inputs_S[1][i].view(-1).tolist())
                    B_log_probas[idx][id_dataset].append(
                        Entry(id_dataset, tuple_msg, log_p.item())
                    )
                    assert(tuple(dataset[id_dataset][0][1][role].tolist()) == entity_vector)
                    # to marginalize over the context, we need to store all
                    # datapoints that share some entity/role pair
                    inputs_entity_role[idx].add((roleset, id_dataset))
                    #  full_idx = (role, full_mat)
                    #  detect_identical[full_idx] += 1
        cond_MI = 0
        n = 0
        max_K = 200
        cond_MI_per_K = defaultdict(list)
        cond_MI_by_idx = defaultdict(list)
        for idx, ids2entries in B_log_probas.items():
            same_entity_role_pair = inputs_entity_role[idx]
            # sanity check
            first_roleset, first_id = list(same_entity_role_pair)[0]
            assert(dataset[first_id][0][0] == first_roleset)
            # end sanity check
            # get ids of datapoints which share the same idx
            all_ids = list(set([j for j in ids2entries.keys()]))
            if len(all_ids) == 1:
                # only the current example i has idx entity/role pair
                # so the log ratio will be = 0
                #  assert(all_ids[0] == i)
                n += 1
                continue
            # find a subset of datapoints sharing entity & role, but not id
            # TODO maybe we should resample everytime?
            K = min(max_K, len(all_ids))
            if len(all_ids) > max_K:
                # too many datapoints, so we subsample
                permutation = torch.randperm(len(all_ids))
                subset_ids = [all_ids[j] for j in permutation[:K]]
            else:
                subset_ids = all_ids
            # prepare inputs to compute denominator (average probability)
            subset_inputs = [dataset[id_][0] for id_ in subset_ids]
            inputs = default_collate(subset_inputs[:K])
            inputs = tuple([e.to(device) for e in inputs])
            log_K = np.log(K)
            #  if idx == (2, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0)):
            #      import pdb; pdb.set_trace()
            for i, list_samples in ids2entries.items():
                cond_MI_by_msg = {}
                cond_MI_buf = 0
                for e in list_samples:
                    # memoization: don't recompute q(m|e,r) for a given m
                    if e.msg not in cond_MI_by_msg:
                        #  for properties in inputs[1]:
                        #      # TODO slow sanity check! add option to enable
                        #      rr, entity_vec = idx
                        #      assert(tuple(properties[rr].tolist()) == entity_vec)
                        msgs = torch.stack([torch.tensor(e.msg),] * K).to(device)  # no torch.tile in 1.7?
                        lp_l = model.eval_proba_sender(inputs, msgs)
                        log_marginal = torch.logsumexp(lp_l, 0) - log_K
                        cond_MI_by_msg[e.msg] = (e.log_p - log_marginal).item()
                    #  print(f"--- log denominator = {torch.logsumexp(lp_l, 0).item()} - {log_K}")
                    # log_ratio = log (q(m|s,e,r) / (1/K Σ q(m|s,e,r)))
                    cond_MI_buf += cond_MI_by_msg[e.msg]
                assert(len(list_samples) == K_MC)
                # at this point, 
                # cond_MI_buf = 1/K_MC Σ_k log_ratio(k)
                cond_MI_buf = cond_MI_buf / float(len(list_samples))
                #  print("- cond_MI_buf", cond_MI_buf)
                cond_MI_per_K[K].append(cond_MI_buf)
                cond_MI_by_idx[idx].append(cond_MI_buf)
                #  print(idx, cond_MI_buf)
                cond_MI += cond_MI_buf
                n += 1

        #  for idx, within_MI in cond_MI_by_idx.items():
        #      #  if idx == (0, (3, 2, 2, 0, 1, 1, 3, 3, 3, 3, 3, 0, 2, 0, 1, 3, 2, 3)):
        #      print(f"I(M;S|e,r={idx})={np.mean(within_MI)} (n={len(within_MI)})")
        cond_MI /= float(n)
        print(f"cond_MI {cond_MI}, n={n}")
        averages = {}
        for k in range(max_K+1):
            if k in cond_MI_per_K:
                mean = np.mean(cond_MI_per_K[k])
                n_k = len(cond_MI_per_K[k])
                print(f"cond_MI_per_K[{k}]={mean} (n={n_k})")
                averages["MI_per_K_" + str(k)] = mean
                averages["n_K_" + str(k)] = n_k
    fn = 'context_ind_' + args.split
    if args.invert_mask:
        fn += '_inv'
    fn += '.json'
    json_fn = os.path.join(dirname, fn)
    with open(json_fn, 'w') as fp:
        data = {"CI": cond_MI}
        data.update(averages)
        print(data)
        json.dump(data, fp)
    exit()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
