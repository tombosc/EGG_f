# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os

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
    # ARCHITECTURE
    parser.add_argument('--sender_nlayers', type=int, default=2)
    parser.add_argument('--receiver_nlayers', type=int, default=2)
    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
    parser.add_argument('--sender_cell', type=str, default='tfm')
    parser.add_argument('--receiver_cell', type=str, default='tfm')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--sender_emb', type=int, default=10,
                        help='Size of the embeddings of Sender (default: 10)')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')

    # OPTIMISATION
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum (β_1 for Adam)")
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help="β_2 for Adam")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1.0)")
    parser.add_argument('--ada_len_cost_thresh', type=float, default=0.0)
    parser.add_argument('--ada_H_cost_thresh', type=float, default=0.0)
    parser.add_argument('--sender_entropy_coeff', type=float, default=0e-2,
                        help="Entropy regularisation coeff for Sender (default: 1e-2)")
    parser.add_argument('--receiver_entropy_coeff', type=float, default=0e-2,
                        help="Entropy regularisation coeff for Receiver (default: 1e-2)")
    parser.add_argument('--distance_reg_coef', type=float, default=0.0,
                        help="To prevent attention from focusing far away from current position")
    parser.add_argument('--entropy_coef', type=float, default=0.)
    parser.add_argument('--length_cost', type=float, default=0.)
    parser.add_argument('--mode', type=str, default='gs',
                        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {rf, gs,"
                             " non_diff} (default: gs)")
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
        receiver_output_objs, labels, entropy_coef, ada_H_cost_thresh):
    """ distrib_message is a list of conditional distributions over
    messages.
    """
    _, labels_objs, _ = labels
    return F.cross_entropy(receiver_output_objs.permute(0,3,1,2), labels_objs,
            reduction="none").sum(1).sum(1)

    #  pred_y = (receiver_output > 0.5).long()
    #  acc = (pred_y == labels).detach().all(dim=1).float()
    #  loss = F.binary_cross_entropy( receiver_output, labels.float(), reduction="none").mean(dim=1)
    #  if distrib_message:
    #      # distrib_message is a (bs, vocab_size) distribution
    #      # if we use variable length messages, it is the distrib of all the
    #      # mini-batch examples over a specific timestep i
    #      # compute empirical marginal q(m_i)
    #      probs_m_i = distrib_message.probs.mean(0)
    #      distr_m_i = Categorical(probs = probs_m_i)
    #      H_m_i = distr_m_i.entropy()#.unsqueeze(0)
    #      # if entropy_coef > 0, marginal entropy is minimized
    #      if ada_H_cost_thresh:
    #          H_coef = loss < ada_H_cost_thresh
    #      else:
    #          H_coef = 1
    #      entropy_penalization = H_coef * entropy_coef * H_m_i
    #      loss += entropy_penalization
    #  else:
    #      # this should be used only with the eos token!
    #      H_m_i = torch.tensor([0.]).squeeze()
    #      entropy_penalization = H_m_i
    #  print("H_m_i size", H_m_i.size())

#  def non_diff_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
#      acc = ((receiver_output > 0.5).long() ==
#             labels).detach().all(dim=1).float()
#      return -acc, {'acc': acc.mean()}

def main(params):
    dataset, opts = get_params(params)
    print(opts)

    device = opts.device

    data_gen = torch.Generator()
    if dataset == 'proto':
        data = DataProto(seed=opts.data.dataset_seed, augment=opts.data.augment, 
                         shuffle_roles=opts.data.shuffle_roles)
    ratios = (opts.data.train_ratio, opts.data.valid_ratio,
              1 - (opts.data.train_ratio + opts.data.valid_ratio))
    # test data will be used to test after model selection
    # prob in another script
    train_data, valid_data, _ = data.random_split(ratios, data_gen)
    torch.manual_seed(opts.random_seed)
    train_loader = DataLoader(train_data, batch_size=opts.batch_size)
            
    valid_loader = DataLoader(valid_data, batch_size=opts.batch_size)
    loss_objs_ = partial(loss_objs,
        entropy_coef=opts.entropy_coef,
        ada_H_cost_thresh = opts.ada_H_cost_thresh,
    )
    if opts.sender_cell == 'tfm' and opts.mode == 'gs':
        predict_roleset = False
        receiver = Receiver(
            dim_emb=opts.sender_emb, dim_ff=opts.sender_hidden,
            vocab_size=opts.vocab_size, dropout=opts.dropout,
            max_len=opts.max_len,
            n_layers=opts.receiver_nlayers,
            distance_reg_coef=opts.distance_reg_coef,
            predict_roleset=predict_roleset,
        )
        sender = Sender(
            dim_emb=opts.sender_emb, dim_ff=opts.sender_hidden,
            vocab_size=opts.vocab_size, dropout=opts.dropout,
            max_len=opts.max_len, 
            n_layers=opts.sender_nlayers,
        )
        sender = TransformerSenderGS(
            agent=sender, vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb, max_len=opts.max_len,
            num_layers=opts.sender_nlayers,
            num_heads=8, hidden_size=opts.sender_hidden,
            temperature=opts.temperature,
            dropout=opts.dropout,
            causal=False,  # causal shouldn't matter, b/c only use the last token
            distance_reg_coef=opts.distance_reg_coef,
        )
        game = SenderReceiverTransformerGS(sender, receiver, 
                loss_roles=loss_roles if predict_roleset else None,
                loss_objs=loss_objs_,
                length_cost = opts.length_cost,
                ada_len_cost_thresh = opts.ada_len_cost_thresh,
        )
    else:
        raise NotImplementedError()

    #  game = core.SenderReceiverRnnReinforce(
    #          sender, receiver, diff_loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=opts.receiver_entropy_coeff)
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
        valid_loader, device=device, is_gs=opts.mode == 'gs',
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
    #  with open(dataset_json_path, 'w') as f:
    #      f.write(opts.data.dumps_json())
    #  callbacks = [entropy_calculator, interaction_saver]#, log_norms, post_train_analysis]
    callbacks = [entropy_calculator, interaction_saver]#, log_norms, post_train_analysis]
    if opts.patience > 0:
        early_stop = EarlyStopperNoImprovement(opts.patience, 'loss')
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
