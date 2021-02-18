# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from simple_parsing import ArgumentParser, Serializable
from dataclasses import dataclass, fields
import numpy as np

import egg.core as core
from egg.core import PrintValidationEvents, get_opts
from .data_readers import data_selector, loaders_from_dataset, dataset_fingerprint
from .architectures import (
    Hyperparameters, EGGParameters, create_game, GlobalParams, RetrainParams
)
from .utils import count_params, set_torch_seed

from .config import (
    compute_exp_dir, save_configs, represent_dict_as_str, load_configs,
)
from .callbacks import (
    InteractionSaver, FileJsonLogger, LRScheduler, CoefScheduler,
)


def set_global_opts(checkpoint_dir, checkpoint_freq):
    opts = get_opts()
    opts.checkpoint_dir = checkpoint_dir
    opts.checkpoint_freq = checkpoint_freq

def get_config(params):
    # this is quite messy, because parsing is in 2 phases:
    # 1. parse GlobalParameters indicating what Dataset class is used, whether
    #    we load a checkpoint, (and possibly in the future: what model is used,
    #    etc.).
    # 2. parse Dataset parameters parsed in phase 1, etc. 
    # it's actually more complex, because when we continue training, we ignore
    # most of the command line arguments, since we read config from the
    # experiment directory.
    # phase 1:
    parser_1 = ArgumentParser()
    parser_1.add_arguments(GlobalParams, dest='glob')
    parser_1.add_argument('--load_from_checkpoint', default='')
    args_1, left_over = parser_1.parse_known_args()
    parser_2 = ArgumentParser()
    checkpoint_fn = args_1.load_from_checkpoint
    if checkpoint_fn:
        # read config files
        if not os.path.exists(checkpoint_fn):
            raise ValueError("Missing checkpoint: {}".format(checkpoint_fn))
        exp_dir = os.path.dirname(checkpoint_fn)
        configs = load_configs(exp_dir)
        data_cls = data_selector[configs['glob'].data]
        print(args_1)
        print("data_cls", data_cls)
        print(configs['data'])
        print(configs['glob'])
        parser_2.add_arguments(RetrainParams, dest='retrain')
    else:
        data_cls = data_selector[args_1.glob.data]
        parser_2.add_arguments(data_cls.Config, dest='data')
    # When we load a checkpoint, we sometimes retrain part of the architecture
    # (see --retrain...). In that case, we want to control the seed. That's why
    # we use Hyperparameters.
    parser_2.add_arguments(Hyperparameters, dest='hp')
    parser_2.add_argument('--print_validation_events', action='store_true')
    args_2 = core.init(parser_2, left_over)
    # merge the namespaces
    args_2.glob = args_1.glob
    if checkpoint_fn:
        configs['hp'].seed = args_2.hp.seed
        configs['retrain'] = args_2.retrain
        # set global opts that are read by core.trainer
        configs['core'].fill_namespace(args_2)
        args_2.load_from_checkpoint = checkpoint_fn  # very important! has
        #  already been parsed during phase 1, but core.init has its own.
    else:
        core_params = EGGParameters.from_argparse(args_2)
        configs = {  # order matters, must match that of load_configs
            'glob': args_2.glob,
            'data': args_2.data,
            'hp': args_2.hp,
            'core': core_params,
        }
    return configs, data_cls, checkpoint_fn
 
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
                  retrain.retrain_receiver_shuffled or
                  retrain.retrain_receiver_deduped)
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

    if (hyper_params.validation_batch_size==0):
        hyper_params.validation_batch_size=get_opts().batch_size
    dataset = data_cls(data_config)
    print("data:", dataset_fingerprint(dataset))

    train_loader, val_loader = loaders_from_dataset(
        dataset,
        data_config,
        seed=data_config.seed,
        train_bs=core_params.batch_size,
        valid_bs=hyper_params.validation_batch_size,
    )

    def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        #  print("sizes msg={}, receiver_in={}, receiv_out={}, lbl={}".format(
        #      _message.size(), _receiver_input.size(), receiver_output.size(),
        #      labels.size()))
        pred = receiver_output.argmax(dim=1)
        acc = (pred == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        n_distractor = dataset.n_distractors(_sender_input)
        return loss, {'acc': acc, 'n_distractor': n_distractor}


    game = create_game(
        core_params,
        data_config, 
        hyper_params,
        loss,
        shuffle_message=retrain.retrain_receiver_shuffled,
        dedup_message=retrain.retrain_receiver_deduped,
    )

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

