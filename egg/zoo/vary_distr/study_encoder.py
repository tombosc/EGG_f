# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from simple_parsing import ArgumentParser, Serializable
from dataclasses import dataclass, fields

import egg.core as core
from egg.core import PrintValidationEvents

import numpy as np

from egg.zoo.vary_distr.utils import count_params
from egg.zoo.vary_distr.data_readers import Data
from egg.zoo.vary_distr.architectures import (
    Hyperparameters, EGGParameters, create_encoder,
)

from .config import compute_exp_dir, save_configs

class EMA():
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha
        self.val = None

    def update(self, v):
        if self.val:
            self.val = self.val * self.alpha + v * (1 - self.alpha)
        else:
            self.val = v

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
    dataset = Data(opts.data, necessary_features=True)

    configs = {
        'data': opts.data,
        'hp': opts.hp,
        'core': core_params,
    }
    print(configs)

    n_combinations = dataset.n_combinations()
    print("#combinations={}".format(n_combinations))
    torch.manual_seed(opts.random_seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
    )

    train_loader = DataLoader(train_ds, batch_size=opts.batch_size,
            shuffle=True, num_workers=1, collate_fn=Data.collater,
            drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=opts.hp.validation_batch_size,
            shuffle=False, num_workers=1, collate_fn=Data.collater,
            drop_last=True,
    )

    model = Predictor(opts.data, opts.hp)
    for n, p in model.named_parameters():
        print("{}: {} @ {}".format(n, p.size(), p.data_ptr()))
    params = list(model.parameters())
    train_params = params
    print("#train_params={}".format(count_params(train_params)))

    optimizer = core.build_optimizer(train_params)
    if (opts.hp.lr_sched == True):
        raise NotImplementedError()
    grad_norm = opts.hp.grad_norm if opts.hp.grad_norm > 0 else None

    criterion = nn.CrossEntropyLoss()
    #  loss_average = EMA(alpha=0.95)
    for i in range(50):
        model.train()
        train_losses = []
        for input_, _, _, n_necessary_features in train_loader:
            optimizer.zero_grad()
            logits = model(input_)
            loss = criterion(logits, n_necessary_features - 1)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print("train{}={}".format(i, np.mean(train_losses)))

        model.eval()
        eval_losses = []
        for input_, _, _, n_necessary_features in val_loader:
            logits = model(input_)
            loss = criterion(logits, n_necessary_features - 1)
            eval_losses.append(loss.item())
        print("eval{}={}".format(i, np.mean(eval_losses)))


class Predictor(nn.Module):
    def __init__(self, data_params, hp):
        super().__init__()
        self.encoder, _ = create_encoder(data_params, hp)
        self.predictor = nn.Sequential(
            nn.Linear(hp.lstm_hidden, hp.lstm_hidden*2),
            nn.ReLU(),
            nn.Linear(hp.lstm_hidden*2, data_params.n_features),
        )

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        logits = self.predictor(encoded)
        return logits

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

