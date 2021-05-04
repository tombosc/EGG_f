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
from .archs import (Receiver, Sender)
from .features import VariableData, FixedData
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from .callbacks import ComputeEntropy, LogNorms, LRAnnealer, PostTrainAnalysis


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
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender (default: 1.0)")
    parser.add_argument('--sender_entropy_coef', type=float, default=0e-2,
                        help="Entropy regularisation coeff for Sender (default: 1e-2)")
    #  parser.add_argument('--receiver_entropy_coeff', type=float, default=0e-2,
    #                      help="Entropy regularisation coeff for Receiver (default: 1e-2)")
    parser.add_argument('--entropy_coef', type=float, default=0.)
    parser.add_argument('--conditional_entropy_coef',
                        default=0.0, type=float)
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
    parser.add_argument('--variable_length',
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
        receiver_output, labels, entropy_coef, conditional_entropy_coef):
    """ distrib_message is a list of conditional distributions over
    messages.
    """
    pred_y = (receiver_output > 0.5).long()
    acc = (pred_y == labels).detach().all(dim=1).float()
    loss = F.binary_cross_entropy( receiver_output, labels.float(), reduction="none").mean(dim=1)
    #  probs = distrib_message.probs
    # we're going to modulate entropy minimization by the cross-entropy loss:
    # if cross-entropy is low, then entropy minimization is high
    # TODO use distrib_message.entropy()?
    #  H = entropy(probs.unsqueeze(0))
    #  H = distrib_message.entropy().unsqueeze(1)
    # in order to get the empirical marginal distrib over messages, we average
    # the conditionalprobabilities:
    empirical_marginal_probs = distrib_message.probs.mean(0)
    marginal = Categorical(probs = empirical_marginal_probs)
    marg_H = marginal.entropy().unsqueeze(0)
    # if entropy_coef > 0, marginal entropy is minimized
    if conditional_entropy_coef:
        # WARNING untested
        entropy_penalization = entropy_coef * (
            marg_H.repeat(pred_y.size(0)) -
            distrib_message.entropy())
        # if this option is set, we also maximize conditional entropy
    else:
        entropy_penalization = entropy_coef * marg_H
    average_H = marg_H.mean()
    loss += entropy_penalization
    logits = distrib_message.logits
    return loss, {'acc': acc, 'H_penal': entropy_penalization, 
            'H_msg': average_H.unsqueeze(0),
            'logits_mean': torch.tensor([logits.mean()]),
            'logits_median': torch.tensor([logits.median()]),
            'logits_min': torch.tensor([logits.min()]),
            'logits_max': torch.tensor([logits.max()]),
    }


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
        vocab_size=opts.vocab_size,
        mlp=opts.sender_mlp,
        predict_temperature=opts.predict_temperature,
        symbol_dropout=opts.symbol_dropout,
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
        conditional_entropy_coef=opts.conditional_entropy_coef,
    )
    if not opts.variable_length:
        sender = create_sender(opts)
        receiver = Receiver(n_bits=opts.n_bits,
                            n_hidden=opts.receiver_hidden,
                            mlp=opts.receiver_mlp)
        if opts.mode == 'gs':
            sender = core.GumbelSoftmaxWrapper(
                agent=sender, temperature=opts.temperature,
                trainable_temperature=opts.gs_train_temperature,
                straight_through=True,
                test_time_sampling=opts.test_time_sampling,
            )
            receiver = core.SymbolReceiverWrapper(
                receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            game = core.SymbolGameGS(sender, receiver, diff_loss)
        elif opts.mode.startswith('rf'):
            if opts.mode == 'rf':
                baseline = MeanBaseline
            elif opts.mode == 'rfn':
                # the baseline will be a copy of the sender, except that it's
                # last output will output a single scalar
                opts_copy = copy.deepcopy(opts)
                opts_copy.vocab_size = 1
                baseline_net = create_sender(opts_copy)
                baseline = lambda: SenderLikeBaseline(baseline_net)
            else:
                raise NotImplementedError()
            sender = core.ReinforceWrapper(agent=sender)  # no baseline here
            receiver = core.SymbolReceiverWrapper(
                receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            receiver = core.ReinforceDeterministicWrapper(agent=receiver)
            game = core.SymbolGameReinforce(
                sender, receiver, diff_loss,
                sender_entropy_coeff=opts.sender_entropy_coeff,
                baseline_type=baseline,
            )
        #  elif opts.mode == 'relax':
        #      sender = core.RelaxSenderWrapper(sender)
        #      receiver = core.SymbolReceiverWrapper(
        #          receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
        #      game = core.RelaxGame(sender, receiver, diff_loss)
    else:
        raise NotImplementedError()
        #  if opts.mode != 'rf':
        #      print('Only mode=rf is supported atm')
        #      opts.mode = 'rf'

        #  if opts.sender_cell == 'transformer':
        #      receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        #      sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
        #                      vocab_size=opts.sender_hidden)  # TODO: not really vocab
        #      sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_emb, max_len=opts.max_len,
        #                                               num_layers=1, num_heads=1, hidden_size=opts.sender_hidden)
        #  else:
        #      receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        #      sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
        #                      vocab_size=opts.sender_hidden)  # TODO: not really vocab
        #      sender = core.RnnSenderReinforce(agent=sender, vocab_size=opts.vocab_size,
        #                                embed_dim=opts.sender_emb, hidden_size=opts.sender_hidden, max_len=opts.max_len, cell=opts.sender_cell)

        #  if opts.receiver_cell == 'transformer':
        #      receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_emb)
        #      receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size, opts.max_len, opts.receiver_emb, num_heads=1, hidden_size=opts.receiver_hidden,
        #                                                       num_layers=1)
        #  else:
        #      receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        #      receiver = core.RnnReceiverDeterministic(
        #          receiver, opts.vocab_size, opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)

        #      game = core.SenderReceiverRnnGS(sender, receiver, diff_loss)

        #  game = core.SenderReceiverRnnReinforce(
        #          sender, receiver, diff_loss, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=opts.receiver_entropy_coeff)
    print(game)
    ####### ATTEMPT 1: penalize fc2 of receiver.
    #  rcv_params = set(game.receiver.agent.fc2[0].parameters()).union(set(game.receiver.agent.fc2[2].parameters()))
    #  rcv_params = set([game.receiver.agent.fc2[2].weight, game.receiver.agent.fc2[0].weight])
    #  other_params = set(game.parameters()) - rcv_params
    #  params = [
    #      {'params': list(rcv_params), 'weight_decay': 0.1},#, 'lr': 1e-4},
    #      {'params': list(other_params)},
    #  ]
    ####### END
    ####### ATTEMPT 2: penalize fc2 of sender to diminish entropy
    #  sdr_params = set([game.sender.agent.fc1[3].weight, game.sender.agent.fc1[3].bias])
    #  other_params = set(game.parameters()) - sdr_params
    #  params = [
    #      {'params': list(sdr_params), 'weight_decay': 0.1},#, 'lr': 1e-4},
    #      {'params': list(other_params)},
    #  ]
    ####### END

    params = list(game.parameters())
    #  if opts.mode == 'rfn':
    #      params += list(baseline_net.parameters())

    if opts.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=opts.lr,
                betas=(opts.momentum, 0.999))
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
        var_length=opts.variable_length, bin_by=bin_by)
    log_norms = LogNorms(game)
    post_train_analysis = PostTrainAnalysis(game)
    callbacks = [entropy_calculator, log_norms, post_train_analysis]
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
