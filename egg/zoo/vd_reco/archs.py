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
        self.emb_column = nn.Linear(n_bits, n_hidden)

        self.fc1 = nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.fc2 = nn.Linear(2 * n_hidden, n_bits)

    def forward(self, embedded_message, bits):
        embedded_bits = self.emb_column(bits.float())

        x = torch.cat([embedded_bits, embedded_message], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        return x.sigmoid()


class ReinforcedReceiver(nn.Module):
    def __init__(self, n_bits, n_hidden):
        super(ReinforcedReceiver, self).__init__()
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
        return sample, log_prob, entropy


class Sender(nn.Module):
    def __init__(self, vocab_size, n_bits, n_hidden,
            predict_temperature=False):
        super(Sender, self).__init__()
        self.emb = nn.Linear(n_bits, n_hidden)
        self.fc1 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden*2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        if predict_temperature:
            self.fc_temperature = nn.Sequential(
                nn.Linear(n_hidden*2, 1),
                nn.Sigmoid(),
            )
        else:
            self.fc_temperature = None 
        #  print(self.fc_temperature.bias)
        self.fc2 = nn.Linear(n_hidden*2, vocab_size)

    def forward(self, bits):
        x = self.emb(bits.float())
        x = F.leaky_relu(x)
        h = self.fc1(x)
        message = self.fc2(h)
        if self.fc_temperature:
            t = self.fc_temperature(h)*2 + 0.2
            message = message / (t + 1e-6)
        #  print(message)
        return message
