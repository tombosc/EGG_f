# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from math import prod
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from simple_parsing import ArgumentParser, Serializable
from dataclasses import dataclass, fields

import egg.core as core
from egg.core import PrintValidationEvents

import numpy as np

from egg.zoo.vary_distr.data_readers import Data
from egg.zoo.vary_distr.architectures import (
    Hyperparameters,EGGParameters, create_game,
)

from .config import compute_exp_dir, save_configs
from .callbacks import InteractionSaver


def exclude_params(parameters, excluded_params):
    other_params = []
    for p in parameters:
        if not any([q.data_ptr() == p.data_ptr() for q in excluded_params]):
            other_params.append(p)
        else:
            print("EXCLUDE", p.data_ptr())
    return other_params


def get_params(params):
    parser = ArgumentParser()
    parser.add_arguments(Data.Config, dest='data')
    parser.add_arguments(Hyperparameters, dest='hp')
    parser.add_argument('--print_validation_events', default=False,
            action='store_true')
    args = core.init(parser, params)
    return args
  


def main(params):
    opts = get_params(params)
    if (opts.hp.validation_batch_size==0):
        opts.hp.validation_batch_size=opts.batch_size

    N = opts.data.n_examples
    train_size = int(N * (3/5))
    val_size = int(N * (1/5))
    test_size = N - train_size - val_size
    # fetch random seed from global params
    opts.data.seed = opts.random_seed
    opts.hp.seed = opts.random_seed
    # in order to keep compatibility with EGG's core (EGG's common CLI), we
    # simply store read parameters in a dataclass.
    core_params = EGGParameters.from_argparse(opts)
    dataset = Data(opts.data)

    configs = {
        'data': opts.data,
        'hp': opts.hp,
        'core': core_params,
    }
    print(configs)

    exps_root = os.environ["EGG_EXPS_ROOT"]
    exp_dir = compute_exp_dir(exps_root, configs)
    save_configs(configs, exp_dir)

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
    val_loader = DataLoader(val_ds, batch_size=opts.hp.validation_batch_size,
            shuffle=False, num_workers=1, collate_fn=collater,
            drop_last=True,
    )
    n_features = dataset.get_n_features()
    embed_dim = opts.hp.embed_dim
    n_features = opts.data.n_features
    max_value = opts.data.max_value

    def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        #  print("sizes msg={}, receiver_in={}, receiv_out={}, lbl={}".format(
        #      _message.size(), _receiver_input.size(), receiver_output.size(),
        #      labels.size()))
        pred = receiver_output.argmax(dim=1)
        acc = (pred == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        n_distractor = dataset.n_distractors(_sender_input)
        #  n_necessary_features = Data.n_necessary_features(_sender_input)
        return loss, {'acc': acc, 'n_distractor': n_distractor}


    game = create_game(core_params, opts.data, opts.hp, loss)

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
    callbacks.append(InteractionSaver(
        exp_dir = exp_dir,
    ))
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

