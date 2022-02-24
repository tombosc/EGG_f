import os
import json
import torch
import argparse
import numpy as np
import copy

import egg.core.util as util
from egg.core.distributed import not_distributed_context

#  from .data_proto import Data as DataProto, init_data as init_data_proto
#  from .archs_protoroles import Hyperparameters, load_game
#  from .train_vl import loss_ordered, loss_unordered

#  def load_model_data_from_cp(checkpoint_fn):
#      """ Load model and data from checkpoint.
#      """
#      dirname = os.path.dirname(checkpoint_fn)
#      dataset_json = os.path.join(dirname, 'data.json')
#      with open(dataset_json, 'r') as f:
#          json_data = json.load(f)
#          dataset_name = json_data["dataset"]
#      checkpoint = torch.load(checkpoint_fn, map_location=torch.device('cpu'))
#      hp_json = os.path.join(dirname, 'hp.json')
#      if dataset_name == 'proto':
#          data_cfg = DataProto.Settings.load(dataset_json)
#          # ignore random seed of the model, b/c will load model
#          dataset, train_data, valid_data, test_data = init_data_proto(data_cfg, 0, 128)
#          hp = Hyperparameters.load(hp_json)
#          if not hp.predict_classical_roles:
#              loss = loss_ordered
#          else:
#              loss = loss_unordered
#          model = load_game(hp, loss, data_cfg.n_thematic_roles)
#          model.load_state_dict(checkpoint.model_state_dict)
#      else:
#          raise ValueError('Unknown dataset', dataset_name)
#      return dirname, hp, model, dataset, train_data, valid_data, test_data

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
