import os
import json
import torch
import argparse
import numpy as np
import copy

import egg.core.util as util
from egg.core.distributed import not_distributed_context
from .data_readers import SimpleData as Data, init_simple_data as init_data
#  from .data_proto import Data as DataProto, init_data as init_data_proto
from .archs import Hyperparameters, load_game
from .train_vl import loss

def load_model_data_from_cp(checkpoint_fn, shuffle_train):
    """ Load model and data from checkpoint.
    """
    dirname = os.path.dirname(checkpoint_fn)
    dataset_json = os.path.join(dirname, 'data.json')
    data_cfg = Data.Settings.load(dataset_json)
    hp_json = os.path.join(dirname, 'hp.json')
    hp = Hyperparameters.load(hp_json)
    dataset, train_loader, valid_loader, test_loader = init_data(data_cfg,
            0, 128, 256, shuffle_train)
    device = 'gpu'
    #  torch.manual_seed(hp.random_seed)  # for model parameters
    checkpoint = torch.load(checkpoint_fn, map_location=torch.device('cpu'))
    print("hp", hp)
    print("data", data_cfg)
    model = load_game(hp, loss, data_cfg)
    model.load_state_dict(checkpoint.model_state_dict)
    return dirname, hp, data_cfg, model, dataset, train_loader, valid_loader, test_loader

def init_common_opts_reloading():
    """ Initialize EGG's global variable to acceptable values for *evaluating*
    models. Not pretty...
    """
    util.common_opts = argparse.Namespace()
    util.common_opts.validation_freq = 0
    util.common_opts.update_freq = 0
    util.no_distributed = True
    util.common_opts.preemptable = False
    util.common_opts.checkpoint_dir = None
    util.common_opts.checkpoint_freq = 0
    util.common_opts.checkpoint_best = ""
    util.common_opts.tensorboard = False
    util.common_opts.fp16 = False
    util.common_opts.load_from_checkpoint = None
    util.common_opts.distributed_context = not_distributed_context()

def simple_classif(model, optimizer, loss_fn, train_dl, valid_dl, test_dl,
                    max_iter, patience=3):
    def eval_(dataloader):
        model.eval()
        losses = []
        for X, y in dataloader:
            y_hat = model(X)
            l = loss_fn(y_hat, y)
            losses.append(l)
        #  print("losses dbg", losses[0], losses[-1])
        return torch.mean(torch.tensor(losses)).item()

    def train_():
        model.train()
        losses = []
        for X, y in train_dl:
            y_hat = model(X)
            l = loss_fn(y_hat, y)
            model.zero_grad()
            l.backward()
            optimizer.step()
            losses.append(l)
        return torch.mean(torch.tensor(losses)).item()

    best_val_loss = float('inf')
    best_epoch = -1
    cur_patience = patience
    train_loss = float('inf')
    for i in range(max_iter):
        val_loss = eval_(valid_dl)
        print(f"Iter {i}: val_loss={val_loss}")
        if val_loss < best_val_loss:
            cur_patience = patience
            best_val_loss = val_loss
            best_epoch = i
            best_epoch_train_loss = train_loss
            best_model = copy.deepcopy(model.state_dict())
        elif cur_patience > 0:
            cur_patience -= 1
        else:
            break
        train_loss = train_()
        print(f"Iter {i}: train_loss={train_loss}")
    model.load_state_dict(best_model)
    assert(eval_(valid_dl) == best_val_loss)
    test_loss = eval_(test_dl)
    return {'val_loss': best_val_loss, 'best_epoch': best_epoch,
            'train_loss': best_epoch_train_loss,
            'test_loss': test_loss}
