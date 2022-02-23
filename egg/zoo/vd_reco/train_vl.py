# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, random_split, BatchSampler, RandomSampler)
from scipy.optimize import linear_sum_assignment

import egg.core as core
from egg.core.smorms3 import SMORMS3
from egg.core import EarlyStopperNoImprovement
from egg.core.baselines import MeanBaseline, SenderLikeBaseline
from .archs_protoroles import Hyperparameters, load_game
from .data_proto import Data as DataProto, init_data as init_data_proto
from .callbacks import ComputeEntropy, LogNorms, LRAnnealer, PostTrainAnalysis, InteractionSaver
from simple_parsing import ArgumentParser

def get_params(params):
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='proto',
                        help="Right now, only 'proto' is supported")
    dataset_args, unknown_args = parser.parse_known_args(params)
    dataset = dataset_args.dataset
    params = unknown_args
    parser = ArgumentParser()
    if dataset == 'proto':
        parser.add_arguments(DataProto.Settings, dest="data")
    else:
        raise ValueError('Unknown dataset')
    # HYPERPARAMETERS + LOSS HYPERPARAMS
    parser.add_arguments(Hyperparameters, dest="hp")
    # OPTIMISATION
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum (β_1 for Adam)")
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help="β_2 for Adam")
    parser.add_argument('--unordered_classical', action='store_true',
                        help='Predict classical roles jointly w/ features')
    #  parser.add_argument('--ada_H_cost_thresh', type=float, default=0.0)
    #  parser.add_argument('--sender_entropy_coeff', type=float, default=0e-2,
    #                      help="Entropy regularisation coeff for Sender (default: 1e-2)")
    #  parser.add_argument('--receiver_entropy_coeff', type=float, default=0e-2,
    #                      help="Entropy regularisation coeff for Receiver (default: 1e-2)")
    #  parser.add_argument('--distance_reg_coef', type=float, default=0.0,
    #                      help="To prevent attention from focusing far away from current position")
    #  parser.add_argument('--entropy_coef', type=float, default=0.)
    #  parser.add_argument('--test_time_sampling', action='store_true', default=False)
    args = core.init(arg_parser=parser, params=params)
    if args.unordered_classical:
        # we have to store it here so that it is saved in the experiment dir.
        # we do not save optimisation hyperparams
        args.hp.predict_classical_roles = True
    else:
        # if the flag is not present, it is still possible that we're reloading
        # a checkpoint containing hp.predict_classical_roles = True!
        # so we shouldn't touch the value here.
        pass
    #  assert args.n_examples_per_epoch % args.batch_size == 0
    return dataset, args

def entropy(probs):
    """ probs is bs, d
    """
    return - (probs.log() * probs).sum(1)


def loss_roles(receiver_output_role, labels):
    labels_role, _, _, _, _ = labels
    return F.cross_entropy(receiver_output_role, labels_role,
            reduction="none")


def loss_objs(_sender_input, _message, receiver_input,
        receiver_output_roles, receiver_output_objs, labels):
    labels_objs = labels[1]
    CE = F.cross_entropy(receiver_output_objs.permute(0,3,1,2), labels_objs,
            reduction="none")
    # dims: batch, obj position, attribute
    # I sum only over attributes and leave the sum on the objects to other code
    # so that I can decompose loss objectwise
    return CE.sum(2)

def loss_ordered(_sender_input, _message, receiver_input,
        receiver_output_roles, receiver_output_objs, labels):
    """ With this loss, the 1st output vector is interpreted as 1st thematic role, 
    2nd vector as 2nd thematic role, etc. So the outputs are ordered according
    to thematic roles, and there is no matching problem. Receiver_output_roles
    is interpreted as a binary prediction of whether the object is missing or
    not. """
    CE = loss_objs(_sender_input, _message, receiver_input,
        receiver_output_roles, receiver_output_objs, labels)
    # only non-missing objects count in the loss
    object_absent = (labels[4] == -1)
    target_object_present = (~ object_absent).long()
    CE = CE.masked_fill(object_absent, 0)
    # prediction of objects which are present vs mere padding 
    CE_present = F.cross_entropy(receiver_output_roles.permute(0,2,1), target_object_present,
            reduction="none")
    return (CE_present, CE)

def loss_classical_roles(receiver_output_roles, labels):
    labels_roles = labels[4] + 1
    CE = F.cross_entropy(receiver_output_roles.permute(0,2,1), labels_roles,
            reduction="none")
    return CE

def loss_unordered(_sender_input, _message, receiver_input,
        receiver_output_roles, receiver_output_objs, labels):
    raise NotImplementedError()  
    # TODO! broken, we need to only match the N (non-padding objects).
    # I'm not sure how to do that yet, but it's prob overkill.
    # match outputs with ground truth, a la GraphVAE:
    # - edges are classical roles
    # - features are the 18 properties
    # we need to match each transformer output to the best ground-truth
    # (node,role) pair.
    # we use the Hungarian algorithm (O(n^3)) to solve that
    # we can compute the loss corresponding to object i being matched to gt j
    # easily: for 3 objects, we will compute a=loss(i=[0, 1, 2]), b=loss(i=[1, 2,
    # 0]), c=loss(i=[2, 0, 1]). 
    # so b[1] corresponds to the loss of object 2 being matched to gt 1.
    # in other words, vertically stacking these matrices give us the cost
    # matrix
    LO = []
    LR = []
    for perm in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
        LO_ = loss_objs(_sender_input, _message, 
                _receiver_input, None, receiver_output_objs[:, perm], labels)
        LR_ = loss_classical_roles(receiver_output_roles[:, perm], labels)
        LO.append(LO_)
        LR.append(LR_)
    # stack and put batch dim first again
    LO = torch.stack(LO).transpose(1, 0)
    LR = torch.stack(LR).transpose(1, 0)
    def normalize(mat):
        return (mat - mat.mean()) / mat.std()

    cost = (LO + LR)
    norm_LO = normalize(LO)
    norm_LR = normalize(LR)
    norm_cost = (norm_LO + norm_LR)
    cpu_norm_cost = norm_cost.cpu().detach().numpy()
    matched_loss_objs, matched_loss_roles = [], []
    for i, c in enumerate(cpu_norm_cost):  # iterate over batch
        row_ind, col_ind = linear_sum_assignment(c)
        matched_loss_obj_ = LO[i, torch.tensor(row_ind), torch.tensor(col_ind)]
        matched_loss_roles_ = LR[i, torch.tensor(row_ind), torch.tensor(col_ind)]
        matched_loss_objs.append(matched_loss_obj_)
        matched_loss_roles.append(matched_loss_roles_)
    return (torch.stack(matched_loss_roles), torch.stack(matched_loss_objs))


def main(params):
    dataset, opts = get_params(params)
    print(opts)

    if dataset == 'proto':
        data, train_loader, valid_loader, _ = init_data_proto(opts.data, opts.random_seed, opts.batch_size)
    device = opts.device
    torch.manual_seed(opts.random_seed)  # for model parameters
    if not opts.hp.predict_classical_roles:
        loss = loss_ordered
    else:
        loss = loss_unordered
    if opts.hp.sender_cell == 'tfm' and opts.hp.mode == 'gs':
        game = load_game(opts.hp, loss, opts.data.n_thematic_roles)
    else:
        raise NotImplementedError()
    print(game)

    params = list(game.parameters())
    #  if opts.mode == 'rfn':
    #      params += list(baseline_net.parameters())

    if opts.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=opts.lr,
                betas=(opts.momentum, opts.adam_beta2))
    elif opts.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=opts.lr, momentum=opts.momentum)
    elif opts.optimizer == 'smorms3':
        optimizer = SMORMS3(params, lr=opts.lr)
    elif opts.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=opts.lr,
                momentum=opts.momentum)

    def bin_by(sender_input_to_send):
        return (sender_input_to_send[1:].sum().item() + 0.5 *
                sender_input_to_send[0].item())


    entropy_calculator = ComputeEntropy(
        valid_loader, device=device, is_gs=opts.hp.mode == 'gs',
        var_length=True, bin_by=bin_by, var_message_length=True)
    log_norms = LogNorms(game)
    post_train_analysis = PostTrainAnalysis(game)
    freq_save = [opts.n_epochs]
    assert(opts.checkpoint_dir)
    interaction_saver = InteractionSaver(
        freq_save, freq_save,
        folder_path=opts.checkpoint_dir,
        save_early_stopping=True,
    )
    # save data config
    os.makedirs(opts.checkpoint_dir)  
    dataset_json_path = os.path.join(opts.checkpoint_dir, 'data.json')
    opts.data.save(dataset_json_path)
    hp_json_path = os.path.join(opts.checkpoint_dir, 'hp.json')
    opts.hp.save(hp_json_path)
    #  with open(dataset_json_path, 'w') as f:
    #      f.write(opts.data.dumps_json())
    #  callbacks = [entropy_calculator, interaction_saver]#, log_norms, post_train_analysis]
    callbacks = [entropy_calculator, interaction_saver]#, log_norms, post_train_analysis]
    if opts.patience > 0:
        best_score_json_path = os.path.join(opts.checkpoint_dir, 'best_scores.json')
        early_stop = EarlyStopperNoImprovement(opts.patience,
                best_score_json_path, True)
        callbacks.append(early_stop)

    if opts.optimizer == 'sgd':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
        callbacks.append(LRAnnealer(scheduler))
    # the logger should be the last one!
    callbacks.append(core.ConsoleLogger(print_train_loss=True, as_json=True))
    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
