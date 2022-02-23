# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import copy
import os

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, random_split, BatchSampler, RandomSampler)
from scipy.optimize import linear_sum_assignment

import egg.core as core
from egg.core.smorms3 import SMORMS3
from egg.core import EarlyStopperNoImprovement
from egg.core.baselines import MeanBaseline, SenderLikeBaseline
from .archs_protoroles import Hyperparameters, load_game
from .data_proto import Data as DataProto, dataset_to_chimeras 
from egg.zoo.language_bottleneck.intervention import CallbackEvaluator
from .callbacks import ComputeEntropy, LogNorms, LRAnnealer, PostTrainAnalysis, InteractionSaver
from .utils import load_model_data_from_cp, init_common_opts_reloading
from simple_parsing import ArgumentParser


def main(params):
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--in-distribution', action='store_true')
    args = parser.parse_args(params)
    dirname, hp, model, dataset, _, valid_loader, test_loader = \
        load_model_data_from_cp(args.checkpoint)
    if hp.version == 1.0:
        # raise error: I've written the script after version 1.1
        # so fail here, it means problem loading the model
        raise NotImplementedError()
    if hp.predict_classical_roles:
        raise NotImplementedError()
    init_common_opts_reloading()

    if args.in_distribution:
        # use the test set
        dummy_loader = valid_loader
        data_loader = test_loader
        json_fn = os.path.join(dirname, 'test_generalization_iid.json')
    else:
        data_loader, dummy_loader = dataset_to_chimeras(dataset, 1024)
        json_fn = os.path.join(dirname, 'test_generalization.json')
    #  device = opts.device
    #  print(f"Device = {device}")

    trainer = core.Trainer(
        game=model, optimizer=None,
        train_data=dummy_loader,
        validation_data=data_loader,
        device='cuda',
    )
    loss_valid, I = trainer.eval()
    # loss_valid is the sum of losses, but we want the reconstruction loss
    # we also decompose the reconstruction loss depending on how many
    # objects are hidden
    loss_reconstruction = I.aux['loss_objs'].mean().item()
    n_to_send = (I.aux['sender_input_to_send'] == 1).sum(1)
    def reconstruction_loss_per_nsend(n):
        right_number_to_send = (n_to_send == n)
        S = I.aux['loss_objs'] * right_number_to_send
        return (S.sum() / right_number_to_send.sum()).item()

    with open(json_fn, 'w') as fp:
        d = {
            'loss_reco': loss_reconstruction,
            'loss_reco_1': reconstruction_loss_per_nsend(1),
            'loss_reco_2': reconstruction_loss_per_nsend(2),
            'loss_reco_3': reconstruction_loss_per_nsend(3),
        }
        json.dump(d, fp)
    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
