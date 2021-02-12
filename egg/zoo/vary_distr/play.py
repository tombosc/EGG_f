# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from functools import reduce 
import operator


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from egg.zoo.vary_distr.utils import count_params

from .config import compute_exp_dir, save_configs
from .callbacks import (InteractionSaver, FileJsonLogger, LRScheduler,
    CoefScheduler)

def prod(iterable):  # python3.7
    return reduce(operator.mul, iterable, 1)


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
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    opts = get_params(params)
    if (opts.hp.validation_batch_size==0):
        opts.hp.validation_batch_size=opts.batch_size
    cp_every_n_epochs = 50
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
    if os.path.exists(exp_dir):
        raise ValueError("Dir already exists: {}".format(exp_dir))
    opts.checkpoint_dir = exp_dir
    opts.checkpoint_freq = cp_every_n_epochs
    save_configs(configs, exp_dir)

    train_loader, val_loader = loaders_from_dataset(
        dataset,
        configs['data'],
        seed=opts.random_seed,
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


    game = create_game(core_params, opts.data, opts.hp, loss)

    params = list(game.parameters())
    for n, p in game.named_parameters():
        print("{}: {} @ {}".format(n, p.size(), p.data_ptr()))
    #  params_no_emb = exclude_params(params, list(shared_encoder.parameters()))
    #  train_params = params_no_emb
    train_params = params

    print("#train_params={}".format(count_params(train_params)))
    optimizer = core.build_optimizer(train_params)
    #  embed_dim = opts.hp.embed_dim
    #  optimizer = get_std_opt(params, opts.hp.embed_dim)

    callbacks = []
    callbacks.append(core.ConsoleLogger(print_train_loss=True, as_json=True))
    callbacks.append(FileJsonLogger(
        exp_dir=exp_dir,
        filename="logs.txt",
        print_train_loss=True
    ))
    callbacks.append(InteractionSaver(
        exp_dir=exp_dir,
        every_epochs=cp_every_n_epochs,
    ))
    if (opts.print_validation_events == True):
        callbacks.append(core.PrintValidationEvents(n_epochs=opts.n_epochs))

    if (opts.hp.lr_sched == True):
        # put that on hold. I have not idea how to tune these params with such
        # small datasets.
        raise NotImplementedError()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)
        callbacks.append(LRScheduler(scheduler))

    if opts.hp.length_coef_epoch > 0:
        sched = CoefScheduler(
            get_coef_fn=game.get_length_cost,
            set_coef_fn=game.set_length_cost,
            init_value=0,
            inc=opts.hp.length_coef / 10.,
            every_n_epochs=opts.hp.length_coef_epoch,
            final_value=opts.hp.length_coef,
        )
        callbacks.append(sched)

    grad_norm = opts.hp.grad_norm if opts.hp.grad_norm > 0 else None

    trainer = core.Trainer(game=game,
                           optimizer=optimizer,
                           train_data=train_loader,
                           validation_data=val_loader,
                           callbacks=callbacks,
                           grad_norm=grad_norm,
    )
    trainer.train(n_epochs=opts.n_epochs)
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

