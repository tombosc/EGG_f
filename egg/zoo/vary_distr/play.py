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
from egg.core import PrintValidationEvents
from .data_readers import data_selector, loaders_from_dataset, dataset_fingerprint
from .architectures import (
    Hyperparameters, EGGParameters, create_game,
)
from .utils import count_params, set_torch_seed

from .config import (
    compute_exp_dir, save_configs, represent_dict_as_str, load_configs,
)
from .callbacks import (
    InteractionSaver, FileJsonLogger, LRScheduler, CoefScheduler,
)

 
@dataclass
class GlobalParams(Serializable):
    data: str = 'id'  
    retrain_receiver: bool = False
    retrain_receiver_shuffled: bool = False
    retrain_receiver_deduped: bool = False

    def __post_init__(self):
        assert(self.data in ['dd', 'id'])
        a = self.retrain_receiver
        b = self.retrain_receiver_shuffled
        c = self.retrain_receiver_deduped
        assert(not(a and b) and not(a and c) and not(b and c))

    def get_dict_dirname(self):
        d = self.__dict__.copy()
        
        def delete_or_replace(a, b):
            if d[a] == False:
                del d[a]
            else:
                d[b] = d[a]
                del d[a]

        delete_or_replace('retrain_receiver_shuffled', 'RSh')
        delete_or_replace('retrain_receiver_deduped', 'RDe')
        delete_or_replace('retrain_receiver', 'R')
        return d


def get_params(params):
    # parsing in 2 phases:
    # 1. parse GlobalParameters indicating what Dataset class is used (and 
    #    possibly in the future, what model is used, etc.).
    # 2. parse Dataset parameters parsed in phase 1, etc. 
    # phase 1:
    parser = ArgumentParser()
    parser.add_arguments(GlobalParams, dest='glob')
    args_phase_1, left_over = parser.parse_known_args()
    data_cls = data_selector[args_phase_1.glob.data]
    # stage 2: parse agent alorithm's options and env params
    parser = ArgumentParser()
    parser.add_arguments(data_cls.Config, dest='data')
    parser.add_arguments(Hyperparameters, dest='hp')
    parser.add_argument('--print_validation_events', action='store_true')
    args = core.init(parser, left_over)
    # merge the namespaces
    args.glob = args_phase_1.glob
    print(args)
    return args, data_cls
 
def main(params):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    cp_every_n_epochs = 50

    opts, data_cls = get_params(params)

    retraining_reicv = (opts.glob.retrain_receiver or 
                  opts.glob.retrain_receiver_shuffled or
                  opts.glob.retrain_receiver_deduped)

    if retraining_reicv: 
        # ignore every parameter, except load_from_checkpoint and seed.
        # from the checkpoint specified in load_from_checkpoint, we're going to
        # read config files.
        checkpoint_dir = opts.load_from_checkpoint
        if not os.path.exists(checkpoint_dir):
            raise ValueError("Missing checkpoint: {}".format(checkpoint_dir))
        exp_dir = os.path.dirname(checkpoint_dir)
        configs = load_configs(exp_dir)
        core_params = configs['core']
        data_config = configs['data']
        hyper_params = configs['hp']
        # then, specify a new directory to save experiments:
        exp_dir = os.path.join(
            exp_dir, represent_dict_as_str(opts.glob) + 'sd_' + str(seed)
        )
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        #  configs['train'] = GlobalParams(opts.glob)
        #  save_configs(configs, exp_dir)
        print(exp_dir)
    else:
        # in order to keep compatibility with EGG's core (EGG's common CLI), we
        # simply store read parameters in a dataclass.
        core_params = EGGParameters.from_argparse(opts)
        data_config = opts.data
        hyper_params = opts.hp
        configs = {
            'glob': opts.glob,
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
        opts.checkpoint_freq = 100
        save_configs(configs, exp_dir)

    if (hyper_params.validation_batch_size==0):
        hyper_params.validation_batch_size=opts.batch_size
    dataset = data_cls(data_config)
    print("data:", dataset_fingerprint(dataset))

    train_loader, val_loader = loaders_from_dataset(
        dataset,
        configs['data'],
        seed=data_config.seed,
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
        return loss, {'acc': acc, 'n_distractor': n_distractor}


    game = create_game(
        core_params,
        data_config, 
        hyper_params,
        loss,
        shuffle_message=opts.glob.retrain_receiver_shuffled,
        dedup_message=opts.glob.retrain_receiver_deduped,
    )

    params = list(game.parameters())
    print("#params={}".format(count_params(params)))

    if retraining_reicv:
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
    if (opts.print_validation_events == True):
        callbacks.append(core.PrintValidationEvents(n_epochs=opts.n_epochs))

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
    if retraining_reicv:
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
        ignore_optimizer_state=retraining_reicv,
    )

    if retraining_reicv:
        game.receiver.load_state_dict(receiver_state)
        first_param_after = game.receiver.state_dict()[name_first_param]
        assert(torch.all(first_param_after == first_param))
    trainer.train(n_epochs=opts.n_epochs)
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

