# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.distributions import Categorical
from functools import partial

import egg.core as core
from egg.core.smorms3 import SMORMS3
from egg.core import EarlyStopperAccuracy
from egg.core.baselines import MeanBaseline, SenderLikeBaseline
from .archs_vl import (Receiver, Sender)
from .gs_wrappers import SenderReceiverRnnGSST
from .features import VariableData, FixedData
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from .callbacks import ComputeEntropy, LogNorms, LRAnnealer, PostTrainAnalysis
from egg.core.callbacks import InteractionSaver


def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_bits', type=int, default=8,
                        help='')
    parser.add_argument('--bits_r', type=int, default=4,
                        help='')
    parser.add_argument('--sender_hidden', type=int, default=10,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=10,
                        help='Size of the hidden layer of Receiver (default: 10)')
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
    parser.add_argument('--entropy_coef', type=float, default=0.)
    parser.add_argument('--length_cost', type=float, default=0.)
    parser.add_argument('--train_test_ratio', type=float, default=-1,
                        help="If -1, train and test are full data.")

    #  parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--mode', type=str, default='gs',
                        help="Selects whether Reinforce or GumbelSoftmax relaxation is used for training {rf, gs,"
                             " non_diff} (default: gs)")
    parser.add_argument('--gs_train_temperature',
                        action='store_true', default=False)
    parser.add_argument('--variable_bits',
                        action='store_true', default=False)
    parser.add_argument('--sender_squash_output',
                        type=float, default=0)
    parser.add_argument('--test_time_sampling', action='store_true', default=False)
    parser.add_argument('--sender_mlp',
                        action='store_true', default=False)
    parser.add_argument('--receiver_mlp',
                        action='store_true', default=False)
    parser.add_argument('--predict_temperature',
                        action='store_true', default=False)
    parser.add_argument('--sender_cell', type=str, default='rnn')
    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--symbol_dropout', type=float, default=0.0)
    parser.add_argument('--sender_emb', type=int, default=10,
                        help='Size of the embeddings of Sender (default: 10)')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')

    args = core.init(arg_parser=parser, params=params)

    #  assert args.n_examples_per_epoch % args.batch_size == 0
    return args

def entropy(probs):
    """ probs is bs, d
    """
    return - (probs.log() * probs).sum(1)


def diff_loss_(_sender_input, _message, distrib_message, _receiver_input,
        receiver_output, labels, entropy_coef, ada_H_cost_thresh):
    """ distrib_message is a list of conditional distributions over
    messages.
    """
    pred_y = (receiver_output > 0.5).long()
    acc = (pred_y == labels).detach().all(dim=1).float()
    loss = F.binary_cross_entropy( receiver_output, labels.float(), reduction="none").mean(dim=1)
    if distrib_message:  
        # distrib_message is a (bs, vocab_size) distribution
        # if we use variable length messages, it is the distrib of all the
        # mini-batch examples over a specific timestep i
        # compute empirical marginal q(m_i)
        probs_m_i = distrib_message.probs.mean(0)  
        distr_m_i = Categorical(probs = probs_m_i)
        H_m_i = distr_m_i.entropy()#.unsqueeze(0)
        # if entropy_coef > 0, marginal entropy is minimized
        if ada_H_cost_thresh:
            H_coef = loss < ada_H_cost_thresh
        else:
            H_coef = 1
        entropy_penalization = H_coef * entropy_coef * H_m_i
        loss += entropy_penalization
    else:
        # this should be used only with the eos token!
        H_m_i = torch.tensor([0.]).squeeze()
        entropy_penalization = H_m_i
    #  print("H_m_i size", H_m_i.size())
    return (loss,
            entropy_penalization, 
            {
                'acc': acc,
            },
            {
                'H_penal': entropy_penalization, 
                'H_m_i': H_m_i,
            })


#  def non_diff_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
#      acc = ((receiver_output > 0.5).long() ==
#             labels).detach().all(dim=1).float()
#      return -acc, {'acc': acc.mean()}

def create_sender(opts):
    if opts.variable_bits:
        n_sender_inputs = opts.n_bits + 1
    else:
        n_sender_inputs = opts.n_bits
    return Sender(n_bits=n_sender_inputs, n_hidden=opts.sender_hidden,
        vocab_size=opts.sender_hidden,  # not vocab size here!
        mlp=opts.sender_mlp,
        predict_temperature=opts.predict_temperature,
        squash_output=opts.sender_squash_output,
    )

def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    if opts.variable_bits:
        if opts.bits_r != 4:
            raise ValueError("These are ignored when --variable_bits")
        data = VariableData(opts.n_bits)
    else:
        data = FixedData(opts.n_bits, opts.bits_r)

    if opts.train_test_ratio != -1:
        train_size = int(opts.train_test_ratio * len(data))
        test_size = len(data) - train_size
        train_data, test_data = random_split(data, (train_size, test_size))
    else:
        train_data = data
        test_data = data
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opts.batch_size)
    diff_loss = partial(diff_loss_,
        entropy_coef=opts.entropy_coef,
        ada_H_cost_thresh = opts.ada_H_cost_thresh,
    )
    #  if opts.mode != 'rf':
    #      print('Only mode=rf is supported atm')
    #      opts.mode = 'rf'
    if opts.sender_cell == 'transformer':
        raise NotImplementedError()
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden,
                mlp=True)
        sender = create_sender(opts)
        sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_emb, max_len=opts.max_len,
                                                 num_layers=1, num_heads=1, hidden_size=opts.sender_hidden)
    else:
        sender = create_sender(opts)
        receiver = Receiver(
            n_bits=opts.n_bits,
            n_hidden=opts.receiver_hidden,
            mlp=opts.receiver_mlp,
        )
        #  receiver = core.SymbolReceiverWrapper(
        #      receiver, vocab_size=opts.vocab_size,
        #      agent_input_size=opts.receiver_hidden,
        #  )
        cell_type='gru'
        if opts.mode == 'gs':
            sender = core.RnnSenderGS(
                sender, 
                opts.vocab_size,
                opts.sender_emb,
                opts.sender_hidden,
                opts.max_len,
                opts.temperature,
                cell=cell_type,
                trainable_temperature=False,
                straight_through=True,
                symbol_dropout=opts.symbol_dropout,
            )
            receiver = core.RnnReceiverGS(
                receiver,
                vocab_size=opts.vocab_size,
                embed_dim=opts.receiver_emb,
                hidden_size=opts.receiver_hidden,
                cell=cell_type,
            )
        elif opts.mode == 'rf':
            raise NotImplementedError()
        #  sender = core.RnnSenderReinforce(agent=sender, vocab_size=opts.vocab_size,
        #                            embed_dim=opts.sender_emb, hidden_size=opts.sender_hidden, max_len=opts.max_len, cell=opts.sender_cell)
    if opts.mode == 'gs':
        game = SenderReceiverRnnGSST(sender, receiver, diff_loss,
                length_cost = opts.length_cost,
                ada_len_cost_thresh = opts.ada_len_cost_thresh,
        )

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
    loss = game.loss

    #  intervention = CallbackEvaluator(test_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=opts.variable_length,
    #                                   input_intervention=True)
    bin_by = 0 if opts.variable_bits else -1
    entropy_calculator = ComputeEntropy(
        test_loader, device=device, is_gs=opts.mode == 'gs',
        var_length=True, bin_by=bin_by, var_message_length=True)
    log_norms = LogNorms(game)
    post_train_analysis = PostTrainAnalysis(game)
    last_epoch = [opts.n_epochs]
    interaction_saver = InteractionSaver(last_epoch, last_epoch)
    callbacks = [entropy_calculator, interaction_saver]#, log_norms, post_train_analysis]
    if opts.optimizer == 'sgd':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
        callbacks.append(LRAnnealer(scheduler))
    # the logger should be the last one!
    callbacks.append(core.ConsoleLogger(print_train_loss=True, as_json=True))
    trainer = core.Trainer(
        game=game, optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=opts.n_epochs)

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
