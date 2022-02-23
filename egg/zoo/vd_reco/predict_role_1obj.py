# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import copy
import os
from collections import Counter, defaultdict
import itertools

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from simple_parsing import ArgumentParser
from egg.core import Trainer
from .utils import (load_model_data_from_cp, init_common_opts_reloading,
        simple_classif)

from .archs_protoroles import FixedPositionalEmbeddings

class ProbeTransformer(nn.Module):
    def __init__(self, dim_emb, dim_ff, vocab_size, n_thematic_roles):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_emb,
            nhead=8,
            dim_feedforward=dim_ff,
            dropout=0.2,
            activation='gelu',
        )  # NOT batch first in 1.7
        self.encoder = nn.TransformerEncoder(encoder_layer, 3)
        self.value_embedding = nn.Embedding(vocab_size, dim_emb)
        self.pos_embedding = FixedPositionalEmbeddings(dim_emb,
                batch_first=True)
        self.predict_role = nn.Linear(dim_emb * 2, n_thematic_roles)

    def forward(self, msg):
        msg_embed = self.value_embedding(msg)
        msg_embed = self.pos_embedding(msg_embed)
        y = self.encoder(msg_embed.transpose(0, 1))  # batch second
        max_y = y.max(0)[0]
        avg_y = y.mean(0)
        features = torch.cat((max_y, avg_y), 1)
        return self.predict_role(features)

class NGramModel(nn.Module):
    def __init__(self, dim_emb, vocab_size, n, n_thematic_roles):
        super().__init__()
        self.n = n
        self.embs = nn.ModuleList([
            nn.Embedding(vocab_size**i, dim_emb) for i in range(1, n+1)
        ])
        self.predict_role = nn.Linear(dim_emb, n_thematic_roles)
        self.factors = nn.Parameter(
            torch.tensor([vocab_size ** i for i in range(0, self.n)]).unsqueeze(0).unsqueeze(0),
            requires_grad=False,
        )
    
    def forward(self, msg):
        bs, max_len = msg.size()
        E = self.embs[0](msg).sum(1)  # embed unigram, sum
        S = max_len  # keep track of how many vectors are summed
        for i, E_mat in enumerate(self.embs[1:], 2):
            for offset in range(i):
                # for example, when msg = [1, 2, 3, 4, 5, 0]
                # i=0: start with [[1,2], [3,4], [5,0]]
                # i=1: start with [[2,3], [4,5]
                m = (max_len - offset) % i
                offsetd = msg[:, offset:max_len-m].reshape(bs, -1, i)
                # represent each n-gram as a unique integer identifier
                offsetd = (offsetd * self.factors[:,:,:i]).sum(2) 
                S += offsetd.size(1) 
                E += E_mat(offsetd).sum(1)
        features = E / S
        return self.predict_role(features)



def main(params):
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--dim_emb', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args(params)
    np.random.seed(0)
    dirname, hp, model, dataset, train_data, valid_data, test_data = \
        load_model_data_from_cp(args.checkpoint)
    init_common_opts_reloading()

    def get_interactions(split):
        # always pass train_data, but we never train, so it's ignored.
        evaluator = Trainer(model, None, train_data, split, 'cpu', None, None, False)
        return evaluator.eval()

    loss_train, I_train = get_interactions(train_data)
    loss_val, I_val = get_interactions(valid_data)
    loss_test, I_test = get_interactions(test_data)

    score_path_json = os.path.join(dirname, 'best_scores.json')
    if os.path.exists(score_path_json):
        with open(score_path_json, 'r') as f:
            best_score = float(json.load(f)['best_score'])
        assert np.abs(loss_val - best_score) < 1e-3, (
            f'Model loading problem? Computed valid score {loss_val} vs'
            f' stored score {best_score}'
        )
    else:
        print("Valid score not found. Couldn't check model loaded correctly.")
    def dloader(data_split, shuffle):
        X = data_split.aux['msg'].detach().to(device).long() 
        to_send = data_split.aux['sender_input_to_send']
        y = (to_send.argmax(1) - 1).to(device)
        return DataLoader(list(zip(X, y)), batch_size=128, shuffle=shuffle,
                drop_last=True)

    def loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    def classif_wrap(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        return simple_classif(model, optimizer, loss,
            dloader(I_train, shuffle=True),
            dloader(I_val, shuffle=False),
            dloader(I_test, shuffle=False),
            max_iter=args.max_iter, patience=args.patience,
        )
 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_ngram_model(n): 
        return NGramModel(
            args.dim_emb,
            vocab_size=hp.vocab_size,
            n=n,
            n_thematic_roles=dataset.n_thematic_roles,
        ).to(device)

    gram1_model = create_ngram_model(1)
    gram1_out = classif_wrap(gram1_model)

    gram2_model = create_ngram_model(2)
    gram2_out = classif_wrap(gram2_model)

    tfm = ProbeTransformer(
        args.dim_emb,
        dim_ff=200,
        vocab_size=hp.vocab_size,
        n_thematic_roles=dataset.n_thematic_roles,
    ).to(device)
    tfm_out = classif_wrap(tfm)

    res = {'tfm': tfm_out, '1gram': gram1_out, 'gram2': gram2_out,
           'patience': args.patience, 'max_iter': args.max_iter,
           'lr': args.lr,
    }
    role_predict_json = os.path.join(dirname, 'role_predict.json')
    with open(role_predict_json, 'w') as f:
        json.dump(res, f)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
