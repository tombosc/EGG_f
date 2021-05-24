# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import egg.core as core


class Receiver(nn.Module):
    def __init__(self, n_bits, n_hidden, mlp):
        super(Receiver, self).__init__()
        self.emb_column = core.RelaxedEmbedding(n_bits, n_hidden)
        self.layer_norm_inp = nn.LayerNorm(n_hidden)
        self.layer_norm_msg = nn.LayerNorm(n_hidden)
        self.mlp = mlp
        if mlp:
            fc1_out = n_hidden * 2
        else:
            fc1_out = n_bits
        self.fc1_message = nn.Linear(n_hidden, fc1_out)
        self.fc1_inputs = nn.Linear(n_hidden, fc1_out)
        if mlp:
            self.fc2 = nn.Sequential(
                nn.ReLU(),
                nn.LayerNorm(fc1_out),
                nn.Linear(fc1_out, n_bits),
            )

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())
        embedded_bits = self.layer_norm_inp(embedded_bits)
        embedded_message = self.layer_norm_msg(embedded_message)
        h1 = self.fc1_inputs(embedded_bits)
        h2 = self.fc1_message(embedded_message)
        h = h1 + h2
        if self.mlp:
            h = self.fc2(h)
        return h.sigmoid()

class Sender(nn.Module):
    def __init__(self, vocab_size, n_bits, n_hidden, mlp,
            predict_temperature=False, 
            squash_output=-1):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_bits, n_hidden)
        self.layer_norm = nn.LayerNorm(n_hidden)
        self.vocab_size = vocab_size
        self.mlp = mlp
        self.squash_output = squash_output
        #  self.fc1 = nn.Linear(n_hidden, vocab_size)
        if mlp:
            fc1_out = n_hidden * 2
        else:
            fc1_out = vocab_size
        self.fc1 = nn.Linear(n_hidden, fc1_out)
        if mlp:
            self.fc2 = nn.Sequential(
                nn.ReLU(),
                nn.LayerNorm(fc1_out),
                nn.Linear(fc1_out, vocab_size),
            )

        if predict_temperature:
            self.fc_temperature = nn.Sequential(  # untested
                nn.Linear(n_hidden, n_hidden*2),
                nn.ReLU(),
                nn.Linear(n_hidden*2, 1),
                nn.ReLU(),
            )
        else:
            self.fc_temperature = None

    def forward(self, bits):
        x = self.emb(bits.float())
        x = self.layer_norm(x)
        h = self.fc1(x)
        if self.mlp:
            h = self.fc2(h)
        if self.fc_temperature:
            t = self.fc_temperature(x)
            print(t.min(), t.max())
            h = h / (t + 0.2)
        #  return h
        if self.squash_output > 0:
            K = self.squash_output
            h = (h.sigmoid() * (2*K)) - K
        return h 