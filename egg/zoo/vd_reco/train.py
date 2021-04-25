# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from functools import partial

import egg.core as core
from egg.core.smorms3 import SMORMS3
from egg.core import EarlyStopperAccuracy
from .archs import (Receiver, ReinforcedReceiver, Sender)
from .features import VariableData, FixedData
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from .callbacks import ComputeEntropy, LogNorms, LRAnnealer


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
    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-2,
                        help="Entropy regularisation coeff for Sender (default: 1e-2)")
    parser.add_argument('--receiver_entropy_coeff', type=float, default=0e-2,
                        help="Entropy regularisation coeff for Receiver (default: 1e-2)")
    parser.add_argument('--entropy_coef', type=float, default=0.)
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
    parser.add_argument('--predict_temperature',
                        action='store_true', default=False)
    parser.add_argument('--variable_length',
                        action='store_true', default=False)
    parser.add_argument('--sender_cell', type=str, default='rnn')
    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--sender_emb', type=int, default=10,
                        help='Size of the embeddings of Sender (default: 10)')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')
    parser.add_argument('--fixed_mlp',
                        action='store_true', default=False)

    args = core.init(arg_parser=parser, params=params)

    #  assert args.n_examples_per_epoch % args.batch_size == 0
    return args

def entropy(probs):
    """ probs is bs, d
    """
    return - (probs.log() * probs).sum(1)


def diff_loss_(_sender_input, _message, distrib_message, _receiver_input,
        receiver_output, labels, entropy_coef):
    pred_y = (receiver_output > 0.5).long()
    acc = (pred_y == labels).detach().all(dim=1).float()
    loss = F.binary_cross_entropy( receiver_output, labels.float(), reduction="none").mean(dim=1)
    #  probs = distrib_message.probs
    thresh = 0.
    if acc.float().mean() > thresh:
        # TODO use distrib_message.entropy()?
        #  H = entropy(probs.unsqueeze(0))
        H = distrib_message.entropy().unsqueeze(0)
        entropy_penalization = entropy_coef * H
    else:
        entropy_penalization = torch.zeros_like(acc)
    loss += entropy_penalization
    return loss, {'acc': acc, 'H_penal': entropy_penalization}


#  def non_diff_loss(_sender_input, _message, _receiver_input, receiver_output, labels):
#      acc = ((receiver_output > 0.5).long() ==
#             labels).detach().all(dim=1).float()
#      return -acc, {'acc': acc.mean()}


def main(params):
    opts = get_params(params)
    print(opts)

    device = opts.device

    if opts.variable_bits:
        if opts.bits_r != 4:
            raise ValueError("These are ignored when --variable_bits")
        data = VariableData(opts.n_bits)
        n_sender_inputs = opts.n_bits + 1
    else:
        data = FixedData(opts.n_bits, opts.bits_r)
        n_sender_inputs = opts.n_bits

    if opts.train_test_ratio != -1:
        train_size = int(opts.train_test_ratio * len(data))
        test_size = len(data) - train_size
        train_data, test_data = random_split(data, (train_size, test_size))
    else:
        train_data = data
        test_data = data
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=opts.batch_size)
    diff_loss = partial(diff_loss_, entropy_coef=opts.entropy_coef)
    if not opts.variable_length:
        sender = Sender(n_bits=n_sender_inputs, n_hidden=opts.sender_hidden,
                        vocab_size=opts.vocab_size,
                        predict_temperature=opts.predict_temperature,
                        fixed_mlp=opts.fixed_mlp,
        )
        receiver = Receiver(n_bits=opts.n_bits,
                            n_hidden=opts.receiver_hidden)
        if opts.mode == 'gs':
            sender = core.GumbelSoftmaxWrapper(
                agent=sender, temperature=opts.temperature,
                trainable_temperature=opts.gs_train_temperature,
                straight_through=True,
            )
            receiver = core.SymbolReceiverWrapper(
                receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            game = core.SymbolGameGS(sender, receiver, diff_loss)
        elif opts.mode == 'rf':
            sender = core.ReinforceWrapper(agent=sender)
            receiver = core.SymbolReceiverWrapper(
                receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            receiver = core.ReinforceDeterministicWrapper(agent=receiver)
            game = core.SymbolGameReinforce(
                sender, receiver, diff_loss, sender_entropy_coeff=opts.sender_entropy_coeff)
        elif opts.mode == 'relax':
            sender = core.RelaxSenderWrapper(sender)  
            receiver = core.SymbolReceiverWrapper(
                receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)
            game = core.RelaxGame(sender, receiver, diff_loss)
        #  elif opts.mode == 'non_diff':
        #      sender = core.ReinforceWrapper(agent=sender)
        #      receiver = ReinforcedReceiver(
        #          n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        #      receiver = core.SymbolReceiverWrapper(
        #          receiver, vocab_size=opts.vocab_size, agent_input_size=opts.receiver_hidden)

        #      game = core.SymbolGameReinforce(sender, receiver, non_diff_loss,
        #                                      sender_entropy_coeff=opts.sender_entropy_coeff,
        #                                      receiver_entropy_coeff=opts.receiver_entropy_coeff)
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
    if opts.optimizer == 'adam':
        optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr,
                betas=(opts.momentum, 0.999))
    elif opts.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(game.parameters(), lr=opts.lr, momentum=opts.momentum)
    elif opts.optimizer == 'smorms3':
        optimizer = SMORMS3(game.parameters(), lr=opts.lr)
    elif opts.optimizer == 'sgd':
        optimizer = torch.optim.SGD(game.parameters(), lr=opts.lr,
                momentum=opts.momentum)
    loss = game.loss

    #  intervention = CallbackEvaluator(test_loader, device=device, is_gs=opts.mode == 'gs', loss=loss, var_length=opts.variable_length,
    #                                   input_intervention=True)
    bin_by = 0 if opts.variable_bits else -1
    entropy_calculator = ComputeEntropy(
        test_loader, device=device, is_gs=opts.mode == 'gs',
        var_length=opts.variable_length, bin_by=bin_by)
    log_norms = LogNorms(game)

    callbacks = [entropy_calculator, log_norms]
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
