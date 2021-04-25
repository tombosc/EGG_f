# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

import egg.core as core


class Receiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(Receiver, self).__init__()
        self.emb_column = core.RelaxedEmbedding(n_bits, n_hidden)
        self.layer_norm_msg = nn.LayerNorm(n_hidden)
        self.layer_norm_inp = nn.LayerNorm(n_hidden)
        self.fc1_message = nn.Linear(n_hidden, n_bits)
        self.fc1_inputs = nn.Linear(n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())
        embedded_bits = self.layer_norm_inp(embedded_bits)
        embedded_message = self.layer_norm_msg(embedded_message)
        h1 = self.fc1_inputs(embedded_bits)
        h2 = self.fc1_message(embedded_message)
        logits = (h1 + h2).sigmoid()
        return logits

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
    def __init__(self, vocab_size, n_bits, n_hidden,
            predict_temperature=False, fixed_mlp=False):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_bits, n_hidden)
        self.layer_norm = nn.LayerNorm(n_hidden)
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(n_hidden, vocab_size)
        #  self.fc1 = nn.Sequential(
        #      nn.Linear(n_hidden, n_hidden),
        #      nn.ReLU(),
        #      nn.LayerNorm(n_hidden),
        #      nn.Linear(n_hidden, vocab_size),
        #  )
        if predict_temperature:
            self.fc_temperature = nn.Sequential(  # untested
                nn.Linear(n_hidden, n_hidden*2),
                nn.ReLU(),
                nn.Linear(n_hidden*2, 1),
                nn.ReLU(),
            )
        else:
            self.fc_temperature = None
        assert(not fixed_mlp)

    def forward(self, bits):
        x = self.emb(bits.float())
        x = self.layer_norm(x)
        h = self.fc1(x)
        if self.fc_temperature:
            t = self.fc_temperature(x)
            print(t.min(), t.max())
            h = h / (t + 0.2)
        return h
