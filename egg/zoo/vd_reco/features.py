# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data


class VariableData(data.Dataset):
    """ Number of bits to show the receiver is variable, but known to the
    sender.
    """
    def __init__(self, n_bits):
        n_examples = 2**(n_bits) 
        numbers = np.array(range(n_examples))
        examples = np.zeros((n_examples, n_bits), dtype=np.int)
        # first dim: indicate n inputs
        sender_inputs = np.zeros((n_examples * (n_bits+1), 1 + n_bits))
        receiver_inputs = np.zeros((n_examples * (n_bits+1), n_bits))

        for i in range(n_bits):
            examples[:, i] = ((np.bitwise_and(numbers, 2 ** i) > 0)*2)-1

        for i in range(n_bits+1):
            beg = i*n_examples
            end = (i+1)*n_examples
            sender_inputs[beg:end, 1:] = examples
            # In order for a linear receiver to work, we need to have bits set
            # to 1 and -1, and missing info set to 0! Otherwise, it needs to be
            # non-linear to be able to interpret receiver_inputs in light of
            # the message.
            receiver_inputs[beg:end] = examples
            # add num bits to transmit information
            sender_inputs[beg:end, 0] = i
            receiver_inputs[beg:end, :i] = 0
        # center num of bits available to receiver input given to sender
        sender_inputs[:, 0] = (2 * sender_inputs[:, 0] / float(n_bits)) - 1.
        self.sender_inputs = sender_inputs
        self.receiver_inputs = receiver_inputs

    def __len__(self):
        return self.sender_inputs.shape[0]

    def __getitem__(self, i):
        labels = (self.sender_inputs[i:, 1:] + 1) / 2
        return self.sender_inputs[i], labels, self.receiver_inputs[i]


class FixedData(data.Dataset):
    """ Number of bits shown to the receiver is fixed.
    """
    def __init__(self, n_bits, n_bits_receiver):
        n_examples = 2**(n_bits) 
        integers = np.array(range(n_examples))
        sender_inputs = np.zeros((n_examples, n_bits))
        receiver_inputs = np.zeros((n_examples, n_bits))

        for i in range(n_bits):
            sender_inputs[:, i] = ((np.bitwise_and(integers, 2 ** i) > 0)*2) - 1
            if i < n_bits_receiver:
                receiver_inputs[:, i] = sender_inputs[:, i]
            # else, stays 0

        self.sender_inputs = sender_inputs
        self.receiver_inputs = receiver_inputs

    def __len__(self):
        return self.sender_inputs.shape[0]

    def __getitem__(self, i):
        labels = (self.sender_inputs[i] + 1) / 2
        return self.sender_inputs[i], labels, self.receiver_inputs[i]
