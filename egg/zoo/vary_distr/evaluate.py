# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, fields

import egg.core as core
from egg.core import PrintValidationEvents

import numpy as np

from .data_readers import loaders_from_dataset, dataset_fingerprint
from .architectures import (
    Hyperparameters, EGGParameters, create_game,
)
from .utils import count_params
from .loss import loss

from .config import compute_exp_dir, load_configs, get_config
from .callbacks import (InteractionSaver, FileJsonLogger, LRScheduler,
    CoefScheduler)

 
def main(params):
    configs, data_cls, checkpoint_fn = get_config(params)
    if not checkpoint_fn:
        raise ValueError()
    print("data_cls", data_cls)
    data_config = configs['data']
    hyper_params = configs['hp']
    core_params = configs['core']
    glob_params = configs['glob']
    retrain = configs['retrain']

    dataset = data_cls(data_config)
    print("data:", dataset_fingerprint(dataset))

    _, val_loader = loaders_from_dataset(
        dataset,
        data_config,
        seed=data_config.seed,
        train_bs=core_params.batch_size,
        valid_bs=hyper_params.validation_batch_size,
    )

    game = create_game(
        core_params,
        data_config, 
        hyper_params,
        loss(dataset),
        shuffle_message=retrain.shuffled,
        dedup_message=retrain.deduped,
    )

    params = list(game.parameters())
    optimizer = core.build_optimizer(params)  # TODO delete?
    #  embed_dim = opts.hp.embed_dim
    #  optimizer = get_std_opt(params, opts.hp.embed_dim)

    callbacks = []
    callbacks.append(core.ConsoleLogger(print_train_loss=True, as_json=True))
    callbacks.append(core.PrintValidationEvents(n_epochs=1))
    trainer = core.Trainer(game=game,
                           optimizer=optimizer,
                           train_data=None,
                           validation_data=val_loader,
                           callbacks=callbacks,
                           grad_norm=None,
                           ignore_optimizer_state=True,
    )

    validation_loss, validation_interaction = trainer.eval()
    for callback in trainer.callbacks:
        callback.on_test_end(
            validation_loss, validation_interaction, 0,
        )  # noqa: E226


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

