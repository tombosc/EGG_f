# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from simple_parsing import ArgumentParser, Serializable
from dataclasses import dataclass, fields

import egg.core as core
from egg.core import PrintValidationEvents

import numpy as np

from egg.zoo.vary_distr.data_readers import Data, loaders_from_dataset
from egg.zoo.vary_distr.architectures import (
    Hyperparameters, EGGParameters, create_game,
)
from egg.zoo.vary_distr.utils import count_params, set_seed

from .config import compute_exp_dir, load_configs
from .callbacks import (InteractionSaver, FileJsonLogger, LRScheduler,
    CoefScheduler)


# Copied from "the annotated transformer"
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def state(self):
      return self.optimizer.state

    @state.setter
    def state(self, new_state):
        self.optimizer.state = new_state

def get_std_opt(params, d_model):
    return NoamOpt(d_model, 2, 4000,
            torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9))

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

    checkpoint_dir = opts.load_from_checkpoint
    if not os.path.exists(checkpoint_dir):
        raise ValueError("Missing checkpoint: {}".format(checkpoint_dir))

    exp_dir = os.path.dirname(checkpoint_dir)

    configs = load_configs(exp_dir)
    core_params = configs['core']
    dataset = Data(configs['data'])
    seed = configs['core'].random_seed
    print("Set seed", seed)
    set_seed(seed)

    train_loader, val_loader = loaders_from_dataset(
        dataset,
        configs['data'],
        seed=seed,
        train_bs=configs['core'].batch_size,
        valid_bs=configs['hp'].validation_batch_size,
    )

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


    game = create_game(core_params, configs['data'], configs['hp'], loss)

    params = list(game.parameters())
    #  for n, p in game.named_parameters():
    #      print("{}: {} @ {}".format(n, p.size(), p.data_ptr()))
    #  params_no_emb = exclude_params(params, list(shared_encoder.parameters()))
    #  train_params = params_no_emb
    train_params = params

    print("#train_params={}".format(count_params(train_params)))
    optimizer = core.build_optimizer(train_params)
    #  embed_dim = opts.hp.embed_dim
    #  optimizer = get_std_opt(params, opts.hp.embed_dim)

    callbacks = []
    callbacks.append(core.ConsoleLogger(print_train_loss=True, as_json=True))
    callbacks.append(core.PrintValidationEvents(n_epochs=1))
    trainer = core.Trainer(game=game,
                           optimizer=optimizer,
                           train_data=train_loader,
                           validation_data=val_loader,
                           callbacks=callbacks,
                           grad_norm=None,
    )

    validation_loss, validation_interaction = trainer.eval()
    for callback in trainer.callbacks:
        callback.on_test_end(
            validation_loss, validation_interaction, 0,
        )  # noqa: E226


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

