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
        self.layer_norm_msg = nn.LayerNorm(n_hidden)
        self.layer_norm_inp = nn.LayerNorm(n_hidden)
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

class ReinforcedReceiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(ReinforcedReceiver, self).__init__()
        raise NotImplementedError()
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())

        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        probs = x.sigmoid()

        distr = Bernoulli(probs=probs)
        entropy = distr.entropy()

        if self.training:
            sample = distr.sample()
        else:
            sample = (probs > 0.5).float()
        log_prob = distr.log_prob(sample).sum(dim=1)
        raise NotImplementedError()
        #  return sample, log_prob, entropy


class Sender(nn.Module):
    def __init__(self, vocab_size, n_bits, n_hidden, mlp,
            predict_temperature=False, symbol_dropout=0.0, 
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
        self.symbol_dropout = symbol_dropout

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
        else:
            K = 100
        if self.training and self.symbol_dropout > 0:
            #  h = h.detach()
            prob = torch.empty(h[:, 0].size()).fill_(self.symbol_dropout)
            mask = torch.bernoulli(prob).byte()
            h[mask, 0] += K
            h[mask, 1:] -= K
            h[mask] = h[mask].detach()
        return h 
