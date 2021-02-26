# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, fields
import numpy as np

import egg.core as core
from egg.core import PrintValidationEvents, get_opts
from .data_readers import data_selector, loaders_from_dataset, dataset_fingerprint
from .architectures import create_game
from .utils import count_params

from .config import (
    compute_exp_dir, save_configs, represent_dict_as_str, load_configs,
    get_config, RetrainParams,
)
from .callbacks import (
    InteractionSaver, FileJsonLogger, LRScheduler, CoefScheduler,
)
from .loss import loss


def set_global_opts(checkpoint_dir, checkpoint_freq):
    opts = get_opts()
    opts.checkpoint_dir = checkpoint_dir
    opts.checkpoint_freq = checkpoint_freq

def main(params):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    cp_every_n_epochs = 50

    configs, data_cls, checkpoint_fn = get_config(params)
    print("data_cls", data_cls)
    data_config = configs['data']
    hyper_params = configs['hp']
    core_params = configs['core']
    glob_params = configs['glob']
    print(glob_params)

    if checkpoint_fn:
        retrain = configs['retrain']
        retraining_recv = (retrain.retrain_receiver or 
                  retrain.shuffled or
                  retrain.deduped)
    else:
        retrain = RetrainParams()
        retraining_recv = False

    # set up model checkpointing and logging
    if retraining_recv: 
        # then, specify a new directory to save logs
        retrain_end_path = represent_dict_as_str(configs['retrain'])
        # write seed, because that param can be changed (unlike others)
        retrain_end_path += 'sd_' + str(hyper_params.seed)
        exp_dir = os.path.dirname(checkpoint_fn)
        exp_dir = os.path.join(exp_dir, retrain_end_path)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        # don't save any config files or checkpoints
        print(exp_dir)
    else:
        exps_root = os.environ["EGG_EXPS_ROOT"]
        exp_dir = compute_exp_dir(exps_root, configs)
        if os.path.exists(exp_dir):
            raise ValueError("Dir already exists: {}".format(exp_dir))
        set_global_opts(checkpoint_dir=exp_dir, checkpoint_freq=100)
        save_configs(configs, exp_dir)

    dataset = data_cls(data_config)
    print("data:", dataset_fingerprint(dataset))

    train_loader, val_loader = loaders_from_dataset(
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
    print("Model")
    print(game)

    params = list(game.parameters())
    print("#params={}".format(count_params(params)))

    if retraining_recv:
        train_params = list(game.receiver.parameters())
    else:
        train_params = params
    for n, p in game.named_parameters():
        print("{}: {} @ {}".format(n, p.size(), p.data_ptr()))

    print("#train_params={}".format(count_params(train_params)))
    optimizer = core.build_optimizer(train_params)

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
    if (get_opts().print_validation_events == True):
        callbacks.append(core.PrintValidationEvents(n_epochs=get_opts().n_epochs))

    if (hyper_params.lr_sched == True):
        # put that on hold. I have not idea how to tune these params with such
        # small datasets.
        raise NotImplementedError()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1.0, gamma=0.95)
        callbacks.append(LRScheduler(scheduler))

    if hyper_params.grad_estim in ['gs', 'gs_st']:
        callbacks.append(
            core.TemperatureUpdater(agent=game.sender,
                decay=hyper_params.gumbel_T_decay, minimum=0.1)
        )


    if hyper_params.length_coef_epoch > 0:
        sched = CoefScheduler(
            get_coef_fn=game.get_length_cost,
            set_coef_fn=game.set_length_cost,
            init_value=0,
            inc=hyper_params.length_coef / 10.,
            every_n_epochs=hyper_params.length_coef_epoch,
            final_value=hyper_params.length_coef,
        )
        callbacks.append(sched)

    grad_norm = hyper_params.grad_norm if hyper_params.grad_norm > 0 else None
    if retraining_recv:
        # to retrain receiver, save untrained receiver's state, load
        # checkpoint, and then load back fresh receiver.
        receiver_state = {
            k: t.clone() for k, t in game.receiver.state_dict().items()}
        # checking that it works by comparing before and after on a random param
        name_first_param, first_param = list(receiver_state.items())[0]

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
        grad_norm=grad_norm,
        ignore_optimizer_state=retraining_recv,
    )

    if retraining_recv:
        game.receiver.load_state_dict(receiver_state)
        first_param_after = game.receiver.state_dict()[name_first_param]
        assert(torch.all(first_param_after == first_param))
    trainer.train(n_epochs=get_opts().n_epochs)
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

