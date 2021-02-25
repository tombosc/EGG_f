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

print(loss)
 
def main(params):
    #  global loss
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

    # creating the trainer is just a hack to get the model to be loaded.
    params = list(game.parameters())
    optimizer = core.build_optimizer(params) 
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

    game.eval()
    for batch in val_loader:
        loss_, interactions = game(*batch, intermediate_predictions=True)
        aux = interactions.aux
        batch_n_objects = dataset.n_distractors(interactions.sender_input).long() + 1
        batch_pred = aux['intermediate_predictions'].transpose(0, 1)
        batch_enc = aux['intermediate_message'].transpose(0, 1)
        print("mean msg len:",
              interactions.message_length.float().mean().item())
        for h, o, n_objects, msg in zip(batch_enc, batch_pred, batch_n_objects,
                interactions.message):
            print("----------------------")
            n_objects = n_objects.item()
            out_probas = F.softmax(o[:, :n_objects], dim=1)
            max_proba = out_probas.max(dim=1).values
            max_idx = out_probas.max(dim=1).indices
            steps_o = step_norms(out_probas)
            steps_h = step_norms(h)
            #  import pdb; pdb.set_trace()
            for letter, step_o, step_h, m, i in zip(msg, steps_o, steps_h, max_proba, max_idx):
                print("{}: {:.3f} ; {:.3f} (p({}) = {:.3f})".format(letter, step_h,
                    step_o, i, m))
            #  break

def step_norms(M):
    """ if M is a matrix, returns O[i] = ||M[i+1] - M[i]||^2, O[0] = 0
    """
    norms = [0.0]
    for i in range(M.size(0)-1):
        sub = M[i+1] - M[i]
        norms.append(torch.dot(sub, sub))
    return torch.tensor(norms) 

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

