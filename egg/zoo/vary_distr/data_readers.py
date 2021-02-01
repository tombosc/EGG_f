# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def strvec2intvec(v):
    return list(map(int, v))


def to_one_hot(l, n_values):
    """ Returns a torch one-hot vector given a python list l.
    """
    v = np.zeros((len(l), n_values), dtype=np.int8)
    v[range(len(l)), l] = 1
    return torch.Tensor(v)


class Data(Dataset):
    """ The dataset is a list of tuples:
    * sender_input: a matrix where each row is a one hot vector. 
    * label: the index of the first row of sender_input in receiver_input 
    * receiver_input: sender_input with shuffled rows.
    To summarize: sender sees sender_input, has to encode some info about its
    first row *relative to* its other rows, receiver sees a shuffled version of
    receiver_input and has to find label. """

    @dataclass
    class Config(Serializable):
        seed: int = 0
        n_examples: int = 1024*5
        min_distractors: int = 1
        max_distractors: int = 4
        max_value: int = 4
        n_features: int = 5
 
    def __init__(self, config):
        c = config
        assert(0 < c.min_distractors < c.max_value)
        self.n_features = c.n_features
        self.max_value = c.max_value
        self.min_distractors = c.min_distractors
        self.max_distractors = c.max_distractors
        self.frame = []
        self.rng = np.random.default_rng(c.seed)

        #  def generate_example():
        #      # TODO is it max_value, or max_value+1 in embedding?
        #      # TODO assert that there can't be 0 distractors, it seems to happen
        #      n_distractors = rng.integers(low=c.min_distractors, high=c.max_distractors)
        #      n_necessary_features = rng.integers(low=c.min_distractors, high=c.max_distractors)
        #      label = rng.choice(n_distractors+1)
        #      features = rng.integers(
        #          low=1,
        #          high=c.max_value+1,
        #          size=(n_distractors+1, c.n_features),
        #      )
        #      return (features, label)


        for i in range(c.n_examples):
            sender_input, n_distractors = self.generate_example()
            # receiver needs shuffled inputs
            permut = self.rng.permutation(np.arange(0, n_distractors+1))
            label = np.argwhere(permut == 0)[0]
            receiver_input = sender_input[permut]
            self.frame.append((
                torch.Tensor(sender_input).long(),
                torch.Tensor(label).long(),
                torch.Tensor(receiver_input).long(),
            ))

    def generate_example(self):
        #  print(n_distractors)
        #  n_necessary_features = self.rng.integers(1, self.n_features+1)
        n_necessary_features = min(self.rng.geometric(p=0.35),
                                   self.n_features)
        max_distractors = min(self.max_distractors,
                              n_necessary_features*(self.max_value)-1)
        #  print(self.min_distractors, max_distractors)
        if self.min_distractors == max_distractors:
            n_distractors = max_distractors
        else:
            n_distractors = self.rng.integers(self.min_distractors, max_distractors)
        #  print(n_necessary_features)
        necessary_features = self.rng.choice(self.n_features,
                replace=False, size=n_necessary_features)
        features = np.zeros((n_distractors + 1, self.n_features))
        features[0] = self.rng.integers(
            low=1,
            high=self.max_value+1,
            size=(self.n_features),
        )
        #  print("necess={}, distract={}".format(n_necessary_features,
        #      n_distractors))
        for i in range(1, n_distractors+1):
            #  print("Loop", i)
            features[i] = self.generate_from(
                features[0],
                features[0:i],
                necessary_features,
            )
        return features, n_distractors

    def generate_from(self, vector, set_vectors, changing_features):
        """ Sample a variant from vector where features #changing_features can
        be modified and different from all vectors in set_vectors.
        """
        new_v = vector.copy()
        def valid_vector(candidate):
            #  print(candidate == set_vectors)
            return ~np.any(np.all(candidate == set_vectors, 1))

        while True:
            self.rng.shuffle(changing_features)
            for j in changing_features:
                new_v[j] = self.rng.integers(1, self.max_value+1)
                if valid_vector(new_v):
                    #  print("Returns", vector, new_v)
                    return new_v


    def difficulty(self, sender_input):
        """ Sender_input is (bs, L_max, n_feat).
        """
        #  print(sender_input[0])
        #  print(sender_input[1])
        tgt = sender_input[:, 0]  # (bs, n_feat)
        dis = sender_input[:, 1:]  # (bs, L_max-1, n_feat)
        is_eq = (tgt.unsqueeze(1) == dis).int() # (bs, L_max-1, n_feat)
        return torch.Tensor([3])

    def n_distractors(self, sender_input):
        return torch.all(sender_input != 0, dim=-1).float().sum(1) - 1

    @classmethod
    def n_necessary_features(cls, sender_input):
        """ Number of features necessary to distinguish the target (first row
        of sender_input) from the distractors.
        """
        n_features = sender_input.size(2)
        n_objs = torch.all(sender_input != 0, dim=-1).long().sum(1)
        mask = torch.all(sender_input == 0, dim=-1).unsqueeze(2)
        diff_to_target = (sender_input != sender_input[:,0].unsqueeze(1))
        diff_to_target = diff_to_target.masked_fill(mask, 0)
        # ignoring first dimension which is batch_size:
        # diff_to_target is a matrix M[i, j] set to 0 iff i is a distractor or
        # if i has the same j-th feature than the first object. 
        # by def, M[0, j] = 0 for all j.
        n_necessary = []	
        for i, M in enumerate(diff_to_target):
            potential_necessary = [n_features]
            for comb in powerset(range(0, n_features)):
                if len(comb) == 0:
                    continue
                comb_tensor = torch.tensor(comb, device=sender_input.device)
                # select specific combination of features, among distractors
                D = M[1:n_objs[i].item()].index_select(1, comb_tensor)
                d = torch.all(torch.any(D, 1), 0)
                if d.item():
                    potential_necessary.append(len(comb))
            n_necessary.append(min(potential_necessary))

        #  diff_to_target = diff_to_target.float().sum(-1)  # sum over features
        #  max_, idx = torch.max(diff_to_target[:, 1:], 1)
        #  return max_
        return torch.tensor(n_necessary).float()

    def get_n_features(self):
        return self.n_features

    def n_combinations(self):
        # example: 3 features, 4 possible values: 4**3
        return self.max_value ** self.n_features

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]


