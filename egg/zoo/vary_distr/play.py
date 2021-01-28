# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from math import prod

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents

import numpy as np

from egg.zoo.vary_distr.data_readers import GeneratedData
from egg.zoo.vary_distr.architectures import (
    PragmaticSimpleSender, DiscriReceiverEmbed,
    SharedSubtractEncoder, PragmaticSimpleReceiver
)

def exclude_params(parameters, excluded_params):
    other_params = []
    for p in parameters:
        if not any([q.data_ptr() == p.data_ptr() for q in excluded_params]):
            other_params.append(p)
        else:
            print("EXCLUDE", p.data_ptr())
    return other_params


# the following section specifies parameters that are specific to our games: we will also inherit the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/master/egg/core/util.py
def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments concerning the input data and how they are processed
    #  parser.add_argument('--train_data', type=str, default=None,
    #                      help='Path to the train data')
    #  parser.add_argument('--validation_data', type=str, default=None,
    #                      help='Path to the validation data')
    parser.add_argument('--validation_batch_size', type=int, default=512)
    parser.add_argument('--length_coeff', type=float, default=0.)
    parser.add_argument('--sender_entropy_coeff', type=float, default=0.,
                        help='Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)')
    #  parser.add_argument('--sender_hidden', type=int, default=32)
    parser.add_argument('--receiver_hidden', type=int, default=30)
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

    N = 1024 * 5
    train_size = int(N * (3/5))
    val_size = int(N * (1/5))
    test_size = N - train_size - val_size
    max_value = 4
    n_features = 5
    embed_dim = 50
    max_distractors = 4
    dataset = GeneratedData(
        N,
        max_value,
        min_distractors=1,
        max_distractors=max_distractors,
        n_features=n_features,
        seed=opts.random_seed,
    )

    def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        #  print("sizes msg={}, receiver_in={}, receiv_out={}, lbl={}".format(
        #      _message.size(), _receiver_input.size(), receiver_output.size(),
        #      labels.size()))
        pred = receiver_output.argmax(dim=1)
        #  print("pred", pred)
        #  print("labl", labels)
        acc = (pred == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        difficulty = dataset.difficulty(_sender_input)
        return loss, {'acc': acc}



    n_combinations = dataset.n_combinations()
    if (n_combinations < N):
        # dataset is too small! in fact we'd like n_combinations to be
        # magnitudes higher than N, not just bigger!
        pass
        #  raise ValueError()
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
        # TODO need to pad to the max ALWAYS
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
    shared_encoder = SharedSubtractEncoder(
        n_features=n_features,
        dim_embed=embed_dim,
        max_value=dataset.max_value,
    )
    sender = PragmaticSimpleSender(
        shared_encoder,
    )
    # wrap the sender
    sender = core.RnnSenderReinforce( 
        sender,
        vocab_size=opts.vocab_size,
        embed_dim=embed_dim,
        #  hidden_size=opts.sender_hidden,
        hidden_size=embed_dim,
        cell='lstm',
        max_len=opts.max_len,
        condition_concat=True,
        always_sample=True,
    )
    # THIS IS BROKEN! the idea was to read the message while attenting it
    #  receiver = PragmaticSimpleReceiver(
    #      input_encoder=shared_encoder,
    #      hidden_size=embed_dim,
    #      max_msg_len=opts.max_len,
    #      n_max_objects=max_distractors,
    #      vocab_size=opts.vocab_size,
    #  )
    # Transformer version
    #  receiver = PragmaticReceiver(
    #          n_features=n_features,
    #          n_hidden=opts.receiver_hidden,
    #          dim_embed=embed_dim,
    #          n_embeddings=max_value,
    #          vocab_size=opts.vocab_size,
    #          max_msg_len = opts.max_len,
    #  )
    # Beginning version
    receiver = DiscriReceiverEmbed(
            n_features=n_features,
            n_hidden=opts.receiver_hidden,
            dim_embed=embed_dim,
            n_embeddings=max_value,
            encoder=shared_encoder,
    )
    receiver = core.RnnReceiverDeterministic(
        receiver,
        vocab_size=opts.vocab_size, embed_dim=embed_dim,
        hidden_size=opts.receiver_hidden, cell='lstm',
    )
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.,
        length_cost=opts.length_coeff,
    )
    #  transformer_params = list(receiver.transformer.parameters())
    #  params = [
    #          {'params': transformer_params, 'lr': 0.0001},
    #          {'params': other_params},
    #  ]
    params = list(game.parameters())
    for n, p in game.named_parameters():
        print("{}: {} @ {}".format(n, p.size(), p.data_ptr()))
    #  params_no_emb = exclude_params(params, list(shared_encoder.parameters()))
    #  train_params = params_no_emb
    train_params = params

    def n_params(params):
        S = 0
        for p in params:
            print(p.size())
            S += prod(p.size())
        return S

    print("#train_params={}".format(n_params(train_params)))

    optimizer = core.build_optimizer(params)

    callbacks = []
    callbacks.append(core.ConsoleLogger(print_train_loss=True, as_json=True))
    if (opts.print_validation_events == True):
        callbacks.append(core.PrintValidationEvents(n_epochs=opts.n_epochs))

    trainer = core.Trainer(game=game, optimizer=optimizer,
                           train_data=train_loader,
                           validation_data=val_loader,
                           callbacks=callbacks)
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

