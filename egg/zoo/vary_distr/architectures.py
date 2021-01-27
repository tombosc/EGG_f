# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch

class PragmaticSenderSimple(nn.Module):
    """ Reads a matrix, encode its first row relative to the other rows.

    It is "Simple", because it only subtracts the average embedding of
    distractors to the target embedding to get probabilities.

    Contrast this with senders which reads a single vector, i.e. encode without
    pragmatic considerations.
    """
    def __init__(self, n_hidden, n_features, dim_embed, max_value):
        super().__init__()
        # as a first approx, embed all different features using same embedding
        self.padding_value = 0
        # since 0 is a padding value, all the actual features go from 1 to
        # max_value+1 (not included)
        self.embeddings = nn.Embedding(max_value+1, dim_embed)
        self.fc1 = nn.Linear(n_features * dim_embed, n_hidden)

    def forward(self, x):
        # TODO clean
        #  print("PragmaSenderSimple", type(x), x.size())
        #  x, n = x_and_n
        bs, max_L, n_features = x.size()
        # create mask set to 1 where iff x[i, j] > n[i] elsewhere 0
        mask = (x == self.padding_value).bool()
        n_distractors = (~mask[:, :, 0]).long().sum(1) # (bs,)
        # visual debug:
        #  print(n)
        #  print("-------------")
        #  print(x[:, :, 0])
        x = self.embeddings(x)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        target = x[:, 0]
        distractors = x[:, 1:]
        # (bs, max_L, n_feat, dim_embed)
        mean_distractors = distractors.sum(1) / n_distractors.view((bs, 1, 1))
        # TODO integrate info of n somehow?
        sub = (target - mean_distractors) # (bs, n_feat, dim_embed)
        sub = sub.view((bs, -1))
        transformed = self.fc1(sub)
        return transformed


class DiscriReceiverEmbed(nn.Module):
    """ A basic discriminative receiver, like DiscriReceiver, but which expect
    integer (long) input to embed, not one-hot encoded.
    """
    def __init__(self, n_features, n_hidden, dim_embed, n_embeddings):
        super().__init__()
        self.padding_value = 0
        self.embeddings = nn.Embedding(
            n_embeddings + 1, dim_embed, padding_idx=self.padding_value,
        )
        self.fc1 = nn.Linear(n_features*dim_embed, n_hidden)

    def forward(self, x, _input):
        # mask: (bs, L) set to False iff all features are padded
        mask = torch.all(_input == self.padding_value, dim=-1)
        _input = self.embeddings(_input)
        bs, n_dist, n_feat, embed_dim = _input.size()
        _input = _input.view((bs, n_dist, -1))
        _input = self.fc1(_input).tanh()
        #  print("mm", _input.size(), x.size())
        dots = torch.matmul(_input, torch.unsqueeze(x, dim=-1)).squeeze()
        dots = dots.masked_fill(mask, -float('inf'))
        return dots
