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
from torch.distributions import Categorical
from functools import partial

import egg.core as core
from egg.core.smorms3 import SMORMS3
from egg.core import EarlyStopperNoImprovement
from egg.core.baselines import MeanBaseline, SenderLikeBaseline
from .archs_protoroles import Hyperparameters, load_game
from .data_proto import Data as DataProto, init_data as init_data_proto
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
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
    #  assert args.n_examples_per_epoch % args.batch_size == 0
    return dataset, args

def entropy(probs):
    """ probs is bs, d
    """
    return - (probs.log() * probs).sum(1)


def loss_roles(receiver_output_role, labels):
    labels_role, _, _ = labels
    return F.cross_entropy(receiver_output_role, labels_role,
            reduction="none")


def loss_objs(_sender_input, _message, distrib_message, _receiver_input,
        receiver_output_objs, labels):
    """ distrib_message is a list of conditional distributions over
    messages.
    """
    labels_objs = labels[1]
    return F.cross_entropy(receiver_output_objs.permute(0,3,1,2), labels_objs,
            reduction="none").sum(1).sum(1)

def main(params):
    dataset, opts = get_params(params)
    print(opts)

    if dataset == 'proto':
        data, train_loader, valid_loader, _ = init_data_proto(opts.data, opts.random_seed, opts.batch_size)
    device = opts.device
    torch.manual_seed(opts.random_seed)  # for model parameters
    if opts.hp.sender_cell == 'tfm' and opts.hp.mode == 'gs':
        game = load_game(opts.hp, loss_objs)
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

    #  intervention = CallbackEvaluator(test_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=opts.variable_length,
    #                                   input_intervention=True)
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
                best_score_json_path, 'loss')
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
