# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents

import numpy as np

from egg.zoo.vary_distr.data_readers import GeneratedData
from egg.zoo.vary_distr.architectures import DiscriReceiverEmbed, PragmaticSenderSimple


# the following section specifies parameters that are specific to our games: we will also inherit the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/master/egg/core/util.py
def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments concerning the input data and how they are processed
    #  parser.add_argument('--train_data', type=str, default=None,
    #                      help='Path to the train data')
    #  parser.add_argument('--validation_data', type=str, default=None,
    #                      help='Path to the validation data')
    parser.add_argument('--validation_batch_size', type=int, default=0)
    parser.add_argument('--mode', type=str, default='rf',
                        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)")
    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-1,
                        help='Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)')
    # arguments concerning the agent architectures
    parser.add_argument('--sender_cell', type=str, default='rnn',
                        help='Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--receiver_cell', type=str, default='rnn',
                        help='Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)')
    parser.add_argument('--sender_hidden', type=int, default=30)
    parser.add_argument('--receiver_hidden', type=int, default=30)
    parser.add_argument('--sender_embedding', type=int, default=10)
    parser.add_argument('--receiver_embedding', type=int, default=10)
    # arguments controlling the script output
    parser.add_argument('--print_validation_events', default=False,
            action='store_true')
    args = core.init(parser,params)
    return args
  

def main(params):
    opts = get_params(params)
    if (opts.validation_batch_size==0):
        opts.validation_batch_size=opts.batch_size
    print(opts, flush=True)

    def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {'acc': acc}

    N = 4096
    train_size = int(N * (3/5))
    val_size = int(N * (1/5))
    test_size = N - train_size - val_size
    max_value = 4
    n_features = 7
    embed_dim = 10
    # TODO WHAT ABT TEST?
    dataset = GeneratedData(N, max_value, 1, embed_dim, n_features, opts.random_seed)
    n_combinations = dataset.n_combinations()
    if (n_combinations < N):
        # dataset is too small! in fact we'd like n_combinations to be
        # magnitudes higher than N, not just bigger!
        raise ValueError()
    print("#combinations={}".format(n_combinations))
    torch.manual_seed(opts.random_seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
    )
    def collater(list_tensors):
        inputs = [e[0] for e in list_tensors]
        tgt_index = torch.cat([e[1] for e in list_tensors])
        outputs = [e[2] for e in list_tensors]
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=0)
        return (padded_inputs, tgt_index, padded_outputs)

    train_loader = DataLoader(train_ds, batch_size=opts.batch_size,
            shuffle=True, num_workers=1, collate_fn=collater,
            drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=opts.validation_batch_size,
            shuffle=False, num_workers=1, collate_fn=collater,
            drop_last=True,
    )
    n_features = dataset.get_n_features()
    receiver = DiscriReceiverEmbed(
            n_features=n_features,
            n_hidden=opts.receiver_hidden,
            dim_embed=embed_dim,
            n_embeddings=max_value,
    )
    # TODO sender should be modified to accept not a vector, but a matrix and
    # an index of a row of that matrix.
    # TODO can sender hidden and receiver_hidden be different?!
    sender = PragmaticSenderSimple(
        n_hidden=opts.sender_hidden,
        n_features=n_features,
        dim_embed=10,
        max_value=dataset.max_value,
    )
    if opts.mode.lower() == 'gs':
        # wrap sender and receiver so that agnostic to GS or reinforce.
        #  sender = core.RnnSenderGS(sender, vocab_size=opts.vocab_size,
        #  embed_dim=opts.sender_embedding, hidden_size=opts.sender_hidden,
        #  cell=opts.sender_cell, max_len=opts.max_len,
        #  temperature=opts.temperature) receiver =
        #  core.RnnReceiverGS(receiver, vocab_size=opts.vocab_size,
        #  embed_dim=opts.receiver_embedding, hidden_size=opts.receiver_hidden,
        #  cell=opts.receiver_cell) game = core.SenderReceiverRnnGS(sender,
        #  receiver, loss)
        #  callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
        raise NotImplementedError()
    else: # NB: any other string than gs will lead to rf training!
        sender = core.RnnSenderReinforce(sender, vocab_size=opts.vocab_size,
                embed_dim=opts.sender_embedding,
                hidden_size=opts.sender_hidden, cell=opts.sender_cell,
                max_len=opts.max_len)
        receiver = core.RnnReceiverDeterministic(receiver,
                vocab_size=opts.vocab_size, embed_dim=opts.receiver_embedding,
                hidden_size=opts.receiver_hidden, cell=opts.receiver_cell)
        game = core.SenderReceiverRnnReinforce(sender, receiver, loss,
                sender_entropy_coeff=opts.sender_entropy_coeff,receiver_entropy_coeff=0)
        callbacks = []
    
    optimizer = core.build_optimizer(game.parameters())
    callbacks.append(core.ConsoleLogger(print_train_loss=True, as_json=True))
    if (opts.print_validation_events == True):
        callbacks.append(core.PrintValidationEvents(n_epochs=opts.n_epochs))
    trainer = core.Trainer(game=game, optimizer=optimizer,
                           train_data=train_loader, validation_data=val_loader,
                           callbacks=callbacks)
    trainer.train(n_epochs=opts.n_epochs)

    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

