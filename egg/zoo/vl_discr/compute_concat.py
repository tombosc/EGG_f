import argparse
import json
import copy
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, random_split, BatchSampler, RandomSampler)
from collections import Counter, defaultdict

import egg.core as core
from .utils import load_model_data_from_cp, init_common_opts_reloading
from .data_readers import SimpleData
from simple_parsing import ArgumentParser


def main(params):
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    args = parser.parse_args(params)
    dirname, hp, data_cfg, model, dataset, train_loader, valid_loader, test_loader = \
        load_model_data_from_cp(args.checkpoint, shuffle_train=False)
    if hp.version == 1.0:
        # raise error: I've written the script after version 1.1
        # so fail here, it means problem loading the model
        raise NotImplementedError()
    init_common_opts_reloading()
    disentangled = data_cfg.disentangled

    dummy_loader = valid_loader
    data_loader = train_loader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = core.Trainer(
        game=model, optimizer=None,
        train_data=dummy_loader,
        validation_data=data_loader,
        device=device,
    )
    loss_train, I = trainer.eval()
    msgs = I.message.argmax(2)
    print("msgs debug")
    for m in msgs:
        print(m)
    print("end debug")
    nec = I.aux['necessary_features'].int()
    # loss_valid is the sum of losses, but we want the reconstruction loss

    # we also decompose the reconstruction loss depending on how many
    # objects are hidden
    # I want to compare the most frequently found subsequence when K=2
    # For each pair, there will be a counter
    # to what we can get when K=1
    # pair: (feature_index, value)

    # pair_to_subsequences: pair → Counter of subsequences
    # useful to approximate marginal posterior over pair for models trained
    # only with K=2
    pair_to_subsequences = defaultdict(Counter)
    # pair_to_message: pair → Counter of (full) messages
    # we can check that concatenability metrics correspond when we use
    # subsequences vs ground-truth K=1 messages
    pair_to_message = defaultdict(Counter)

    def len_msg(msg):
        for i, symbol in enumerate(msg):
            if symbol == 0:
                return i

    def get_subsequences(msg, n):
        subsequences = []
        i = len_msg(msg)
        for n_sub in range(1, n+1):
            for begin in range(0, i - n_sub):
                sub = msg[begin:begin+n_sub]
                padded_sub = sub + (0,) * (len(msg) - n_sub)
                subsequences.append(padded_sub)
        return subsequences

    print(get_subsequences((1, 2, 3, 4, 0, 0), 3))
    #  def pair_to_int(pair):
    #      feat, val = pair
    #      return (feat * data_cfg.max_value) + val

    for input_, nec_features, msg in zip(I.sender_input, nec, msgs):
        msg_tuple = tuple(msg.tolist())
        if len_msg(msg_tuple) == 0:
            continue
        values = [input_[0][f].item() for f in nec_features]
        pair1 = (nec_features[0].item(), values[0])
        pair2 = (nec_features[1].item(), values[1])
        K = (nec_features != -1).sum().item()
        if K == 2:
            for pair in [pair1, pair2]:
                i_feat, val = pair
                if i_feat != -1:
                    subsequences = get_subsequences(msg_tuple, 2)
                    pair_to_subsequences[pair].update(subsequences)
        elif K == 1:
            i_feat, val = pair1  # pair2 contains the non -1 feature
            assert(i_feat != -1)
            pair_to_message[pair1][msg_tuple] += 1


    def sample_from_counter(c, n):
        """ Sample from a counter of counts (normalize values & return keys).
        """
        msgs, counts = zip(*list(c.items()))  # unzip
        n_counts = float(sum(counts))
        norm_counts = [c/n_counts for c in counts]
        idx = np.random.choice(len(norm_counts), p=norm_counts, size=n)
        return [msgs[i] for i in idx]

    def concat_msgs(m1, m2):
        n = len(m1)
        concat = m1[:len_msg(m1)] + m2[:len_msg(m2)]
        if len(concat) < n:
            concat += (0,) * (n - len(concat))
        return concat

    def concat_from_counts(C1, C2, n_sample):
        concatenated = []
        msgs_p1 = sample_from_counter(C1, n_sample)
        msgs_p2 = sample_from_counter(C2, n_sample)
        found_empty = False
        for m1, m2 in zip(msgs_p1, msgs_p2):
            #  print("m1, m2", m1, ",", m2, len_msg(m1), len_msg(m2))
            if len_msg(m1) > 0 and len_msg(m2) > 0:
                m1m2 = concat_msgs(m1, m2)
                m2m1 = concat_msgs(m2, m1)
            else:
                raise NotImplementedError("There are empty messages")
            concatenated += [m1m2, m2m1]
        return concatenated

    def compute_concatenations(pair_counter, p1, p2, n_sample):
        C_pair_1 = pair_counter[p1]
        C_pair_2 = pair_counter[p2]
        if len(C_pair_1) == 0 or len(C_pair_2) == 0:
            return []
        return concat_from_counts(C_pair_1, C_pair_2, n_sample)

    # go over training set
    # for each K=2, try to extract relevant K=1
    augmented_msgs = []
    ids = I.aux['id'].int()
    n_approx = 3
    n_exact = 1
    for id_, input_, nec_features, msg in zip(ids, I.sender_input, nec, msgs):
        K = (nec_features != -1).sum().item()
        if K == 1:
            continue
        msg_tuple = tuple(msg.tolist())
        values = [input_[0][f].item() for f in nec_features]
        pair1 = (nec_features[0].item(), values[0])
        pair2 = (nec_features[1].item(), values[1])
        cc = [msg_tuple]
        cc2 = compute_concatenations(pair_to_subsequences, pair1, pair2, n_approx)
        if len(cc2) == 0:
            continue
        cc += cc2
        if disentangled:
            cc3 = compute_concatenations(pair_to_message, pair1, pair2, n_exact)
            if len(cc3) == 0:
                continue
            cc += cc3
        augmented_msgs.append((id_, cc))

    if disentangled:
        N = 1 + n_approx * 2 + n_exact * 2
    else:
        N = 1 + n_approx * 2
    # we "duplicate" each augmented example in order to batch computations
    data = []
    for id_, msgs in augmented_msgs:
        datapoint = dataset[id_]
        sender_in, labels, receiver_in = datapoint
        assert(len(msgs)  == N)
        #  necessary_features = torch.tensor(labels[3])
        for msg in msgs:
            msg = torch.tensor(msg).unsqueeze(0)
            data.append((sender_in, labels, receiver_in, msg))
            #  data.append((sender_in, necessary_features, labels, receiver_in, msg))

    def collater(list_tensors):
        # will simply ignore messages!
        collated_inputs = SimpleData.collater(list_tensors)  
        msgs = torch.cat([e[3] for e in list_tensors], axis=0)
        return collated_inputs + (msgs,)

    def recursive_to(list_tensors, device):
        out = []
        if type(list_tensors) != tuple:
            return list_tensors.to(device)
        for e in list_tensors:
            if type(e) == tuple:
                out.append(recursive_to(e, device))
            else:
                out.append(e.to(device))
        return out
    
    # cannot use the collater! the data is different since it also carries
    # messages
    data = DataLoader(data, shuffle=False, batch_size=N*128,
            collate_fn=collater)
    with torch.no_grad():
        #  for sender_in_obj, nec_features, labels, receiver_in, msgs in data:
        all_diffL, all_concatL, all_refL, all_concatL_approx = [], [], [], []
        all_diffP, all_concatP, all_refP, all_concatP_approx = [], [], [], []
        for d in data:
            d = recursive_to(d, device)
            sender_in, labels, receiver_in, msgs = d
            probas = model.eval_proba_sender(sender_in, msgs, has_max_len=False)
            losses = model.eval_loss_receiver(sender_in, labels, receiver_in, msgs)
            loss = losses['objs']
            # first message is the actually sent message: "reference" loss
            ref_proba = probas[0::N]
            ref_loss = loss[0::N]
            all_refL.append(ref_loss)
            all_refP.append(ref_proba)
            # then come the approximations, in order to compute an oracle: find
            # the lowest loss that come from various possible concatenated msgs
            approx_ids = range(1, n_approx * 2+1)  
            approx_proba = [probas[i::N] for i in approx_ids]
            approx_loss = [loss[i::N] for i in approx_ids]
            best_approx_proba = torch.cat([e.unsqueeze(1) for e in approx_proba], 1).max(1)[0]
            best_approx_loss = torch.cat([e.unsqueeze(1) for e in approx_loss], 1).min(1)[0]
            # these quantities are POSITIVE if approx is better than reference
            # ~p - p > 0 ↔ ~p > p (ref proba is lower, ie worse)
            all_concatP_approx.append(best_approx_proba - ref_proba)
            # l - ~l > 0 ↔ l > ~l (ref loss is higher, ie worse)
            all_concatL_approx.append(ref_loss - best_approx_loss)
            if disentangled:
                # when it's disentangled, we have access to messages obtained
                # using a *single* necessary feature in the mask (K=1)
                l12, l21 = loss[N-2::N], loss[N-1::N]
                p12, p21 = probas[N-2::N], probas[N-1::N]
                # we take the concatenation order that minimizes it (we don't
                # know which, a priori)
                min_12L = torch.cat((l12.unsqueeze(1), l21.unsqueeze(1)), axis=1).min(1)[0]
                max_12P = torch.cat((p12.unsqueeze(1), p21.unsqueeze(1)), axis=1).max(1)[0]
                # same as above, quantities are >0 if concat better than ref!
                all_concatL.append(ref_loss - min_12L)
                all_concatP.append(max_12P - ref_proba)
                # we can also compute how much the approximation of the
                # posterior approx the K=1 method where we have access to
                # feature-value pairs *in isolation*
                # when >0, these quantities say that approx is BETTER
                # >0 ↔ l12 > l12approx: approx is better
                all_diffL.append(min_12L - best_approx_loss)
                all_diffP.append(best_approx_proba - max_12P)
        approx_concatL = torch.cat(all_concatL_approx).mean().item()
        ref_L = torch.cat(all_refL).mean().item()
        approx_concatP = torch.cat(all_concatP_approx).mean().item()
        ref_P = torch.cat(all_refP).mean().item()
        #  print("concatL_approx=", approx_concatL)
        #  print("ref_L=", ref_L)
        #  print("concatP_approx=", approx_concatP)
        #  print("ref_P=", ref_L)
        n = torch.cat(all_refL).size(0)
        res = {
            "n": n,
            "approx_concatL": approx_concatL,
            "approx_concatP": approx_concatP,
            "ref_L": ref_L,
            "ref_P": ref_P,
            "n_approx": n_approx,
        }
        if disentangled:
            approx_vs_gt_L = torch.cat(all_diffL).mean().item()
            concatL = torch.cat(all_concatL).mean().item()
            approx_vs_gt_P = torch.cat(all_diffP).mean().item()
            concatP = torch.cat(all_concatP).mean().item()
            res.update({
                'concatL': concatL,
                'gt_minus_best_approx_L': approx_vs_gt_L,
                'concatP': concatP,
                'gt_minus_best_approx_P': approx_vs_gt_P,
            })
            #  print("diff: best K+1 - best appr =", approx_vs_gt_L)
            #  print("concatL =", concatL)
        print(res)

    with open(os.path.join(dirname, 'analysis.json'), 'w') as f:
        json.dump(res, f)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
