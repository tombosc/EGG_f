import argparse
import json
import copy
import os

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, random_split, BatchSampler, RandomSampler)
from collections import Counter, defaultdict
from dit import Distribution
from dit.shannon import conditional_entropy, mutual_information, entropy

import egg.core as core
from .utils import load_model_data_from_cp, init_common_opts_reloading
from simple_parsing import ArgumentParser
from operator import itemgetter


def main(params):
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    args = parser.parse_args(params)
    dirname, hp, data_cfg, model, dataset, _, valid_loader, _ = \
        load_model_data_from_cp(args.checkpoint, shuffle_train=False)
    init_common_opts_reloading()
    res = {}
    suffix = {True: '_hard', False: '_easy'}
    for hard in [True, False]:
        ood_data = dataset.sample_ood(500, hard=hard)
        ood_loader = DataLoader(
            ood_data,
            batch_size=512,
            shuffle=False, num_workers=0, 
            collate_fn=dataset.collater,
            drop_last=False,
        )
        trainer = core.Trainer(
            game=model, optimizer=None,
            train_data=valid_loader,
            validation_data=ood_loader,  # we're only going to eval anyways
            device='cuda',
        )
        loss, I = trainer.eval()
        res['acc' + suffix[hard]] = I.aux['acc'].mean().item()
        res['ref_L' + suffix[hard]] = I.aux['loss_objs'].mean().item()
    print(res)
    with open(os.path.join(dirname, 'ood_scores.json'), 'w') as f:
        json.dump(res, f)
    core.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
