# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from torch.utils.data import DataLoader
from itertools import chain, combinations, product
import scipy.stats
import unittest
#  from .utils import set_torch_seed


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

def range_except(N, E, start):
    r = range(start, N+start)
    r = [e for e in r if e not in E]
    assert(len(r) == N-len(E))
    return r

class SimpleData(Dataset):
    @dataclass
    class Settings(Serializable):
        seed: int = 0
        n_examples: int = 1000
        max_value: int = 8
        n_features: int = 5
        disentangled: bool = True
        max_distractors: int = 4  # hardcoded, depends on disentangled...
        ood: bool = True  # save some combinations for OoD (see code)
        data_version: float = 1.0
    
    def sample_objects(self, rng, N_distr, ood, ood_hard=False):
        zero = np.zeros(self.n_features, dtype=int)
        # sample an object uniformly
        x_1 = rng.choice(self.max_value, size=self.n_features) + 1
        prev = x_1
        # to sample distractors, simply modify a random feature 
        distractors = []
        modified_features = []
        for e in range(N_distr):
            distractor = prev.copy()
            if ood and ood_hard and e == 0:
                # example: 3 objects
                # 1 1 1 1
                # 2 1 1 1 ← target in hard mode: need to transmit
                # 2 1 2 1   feature 0 along with feature 2 here.
                i_feat = 0
            elif ood and not ood_hard and (e in [0, 1]):
                if e == 0:
                    i_feat = 1
                elif e == 1:
                    i_feat = 2
            else:
                possible_features = range_except(self.n_features,
                        modified_features, start=0)
                i_feat = rng.choice(possible_features)
            modified_features.append(i_feat)
            #  print(f"Sel feature = {i_feat} with val {prev[i_feat]}")
            possible_values = range_except(self.max_value, [prev[i_feat]],
                    start=1)
            distractor[i_feat] = rng.choice(possible_values)
            #  print(f"Chosen val = {distractor[i_feat]}")
            distractors.append(distractor)
            prev = distractor
        #  import pdb; pdb.set_trace()
        objects = [x_1] + distractors
        # add padding
        objects += [zero,]*(self.max_distractors - N_distr)
        objects = np.asarray(objects)
        return objects

    def sample_ood(self, n_examples, hard):
        assert(self.ood)
        rng = np.random.default_rng(self.seed)
        n_distractors = rng.choice(self.max_distractors, 
                                   p=self.vector_proba_ood,
                                   size=n_examples) + 1
        objects = [self.sample_objects(rng, i, ood=True, ood_hard=hard) for i in n_distractors]
        data = []
        for id_, (N, o) in enumerate(zip(n_distractors, objects)):
            # in hard mode, all the targets are the 2nd objects
            sender_input = np.vstack((o[1][np.newaxis, :],
                o[0][np.newaxis, :], o[2:]))
            permut = rng.permutation(self.max_distractors + 1)
            i_target = np.where(permut == 1)[0][0]
            K2, necessary_features = get_necessary_features(torch.tensor(sender_input).unsqueeze(0))
            labels = (np.asarray([2]), np.asarray([N]), np.asarray([i_target]),
                np.asarray(necessary_features[0]), id_)
            data.append((sender_input, labels, o[permut]))
        return data


    @staticmethod
    def is_ood(necessary_features):
        nec = tuple([n for n in necessary_features if n != -1])
        if len(nec) == 2 and 0 in nec:
            return True
        if len(nec) == 2 and (1 in nec and 2 in nec):
            return True
        return False


    def __init__(self, config):
        c = config
        self.data = []
        self.seed = c.seed
        rng = np.random.default_rng(c.seed)
        self.n_features = c.n_features
        self.max_value = c.max_value
        self.max_distractors = c.max_distractors
        self.ood = c.ood
        if self.ood:
            assert(self.max_value >= 4 and self.n_features >= 5)
        # N_distractors: N
        # N_properties needed to distinguish = K
        
        if c.data_version == 1.0:
            assert(c.max_distractors == 4)
            self.vector_proba_ood = np.asarray([0., 1/4., 1/4., 2/4.])
            if c.disentangled:
                self.vector_proba = np.asarray([1/8., 1/8., 1/4., 2/4.])
            else:
                assert(c.n_features >= 5)
                self.vector_proba = np.asarray([0., 1/4., 1/4., 2/4.])
        elif c.data_version == 1.1:
            assert(c.max_distractors == 5)
            self.vector_proba_ood = np.asarray([0., 1/4., 1/4., 1/4., 1/4.])
            if c.disentangled:
                self.vector_proba = np.asarray([1/8., 1/8., 1/4., 1/4., 1/4.])
            else:
                assert(c.n_features >= 5)
                self.vector_proba = np.asarray([0., 1/4., 1/4., 1/4., 1/4.])
        else: 
            raise NotImplementedError()

        prob_K_1 = np.asarray([2/2., 2/3., 2/4., 2/5., 2/6.])
        p_1 = np.dot(self.vector_proba, prob_K_1[:c.max_distractors])
        print(f"p(K=1)={p_1}, p(K=2)={1-p_1}")

        n_distractors = rng.choice(c.max_distractors, p=self.vector_proba, size=c.n_examples) + 1
        dict_K = {  # maps (N+1, i) to how many features are needed
            (2, 0): 1, (2, 1): 1,
            (3, 0): 1, (3, 1): 2, (3, 2): 1,
            (4, 0): 1, (4, 1): 2, (4, 2): 2, (4,3): 1,
            (5, 0): 1, (5, 1): 2, (5, 2): 2, (5,3): 2, (5,4): 1,
            (6, 0): 1, (6, 1): 2, (6, 2): 2, (6,3): 2, (6,4): 2, (6,5): 1,
        }
        for id_, N in enumerate(n_distractors):
            while True:
                objects = self.sample_objects(rng, N, ood=False)
                # select target randomly
                # this might need to be repeated, in case we do OoD, since we don't
                # want "special" OoD samples in the train set 
                # (the OoD test sets are generated separately)
                patience = 3
                while True:
                    if c.disentangled:
                            i_target = rng.choice(N+1)
                            K = dict_K[(N+1, i_target)]
                    else:
                        # if entangled, never choose first or last
                        i_target = rng.choice(range(1, N))  
                        K = dict_K[(N+1, i_target)]
                        assert(K > 1)
                    sender_input = np.vstack((
                        objects[i_target][np.newaxis, :], 
                        objects[:i_target],
                        objects[i_target+1:]
                    ))
                    assert(np.allclose(sender_input[0], objects[i_target]))
                    K2, necessary_features = get_necessary_features(torch.tensor(sender_input).unsqueeze(0))
                    necessary_features = necessary_features[0]  # since we don't pass a batch
                    #  print("nec", necessary_features)
                    if not self.ood or (not self.is_ood(necessary_features)):
                        break
                    elif patience >= 1:
                        patience -= 1
                    else:
                        break
                if patience != 0:
                    #  print("no more patience, regenerate")
                    # when we break the loop here, we can finish the procedure
                    # and the object is not ood. else, we regenerate. 
                    break
            if K2 == 1:
                necessary_features = necessary_features + (-1,)  # padding
            # verify that the computation of necessary feature yields at least
            # the same number of necessary features as was planned
            assert(K == K2.item())
            labels = (
                np.asarray([K]), np.asarray([N]),
                np.asarray([i_target]),
                np.asarray(necessary_features),
                id_,
            )
            receiver_input = objects
            self.data.append((sender_input, labels, receiver_input))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @staticmethod
    def collater(list_tensors):
        sender_inputs = [e[0] for e in list_tensors]
        #  labels = torch.cat([e[1] for e in list_tensors])
        receiver_inputs = [e[2] for e in list_tensors]
        p_sender_i = torch.tensor(sender_inputs)
        p_receiver_i = torch.tensor(receiver_inputs)
        K = torch.tensor([e[1][0] for e in list_tensors]).squeeze()
        N = torch.tensor([e[1][1] for e in list_tensors]).squeeze()
        i_target = torch.tensor([e[1][2] for e in list_tensors]).squeeze()
        nec_features = torch.stack([torch.tensor(e[1][3]) for e in list_tensors]).squeeze()
        ids = torch.tensor([e[1][4] for e in list_tensors]).squeeze()
        #  padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        #  padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=0)
        return ((p_sender_i, nec_features),
            (K, N, i_target, nec_features, ids),
            p_receiver_i,
        )

class TesterData(unittest.TestCase):
    def test_ood_data(self):
        print("OoD test")
        c = SimpleData.Settings(n_examples=10, max_value=10, ood=True)
        data = SimpleData(c)
        dataset = data.sample_ood(100, hard=True)
        for sender_in, labels, recv_in in dataset:
            K, N, target, nec_features, x = labels
            assert(nec_features[0] == 0)
            assert(np.allclose(sender_in[0], recv_in[target]))
            assert(data.is_ood(nec_features))
        dataset = data.sample_ood(100, hard=False)
        for sender_in, labels, recv_in in dataset:
            K, N, target, nec_features, x = labels
            assert(nec_features[0] == 1 and nec_features[1] == 2)
            assert(np.allclose(sender_in[0], recv_in[target]))
            assert(data.is_ood(nec_features))

    def test_old_simple_data(self):
        print("Simple data test")
        c = SimpleData.Settings(n_examples=3000, max_value=8,
                disentangled=True, ood=True)
        data = SimpleData(c)
        print("Fingerprint", dataset_fingerprint(data))
        count_K = Counter()
        for sender_in, labels, recv_in in data.data:
            K, N, target, nec_features, x = labels
            count_K[K[0]] += 1
            #  print(sender_in[0], nec_features)
            assert(np.allclose(sender_in[0], recv_in[target]))
            assert(not data.ood or not data.is_ood(nec_features))
        print(count_K)

    def test_simple_data(self):
        print("Simple data test")
        c = SimpleData.Settings(n_examples=3000, max_value=10,
                n_features=7, max_distractors=5, data_version=1.1,
                disentangled=True, ood=True)
        data = SimpleData(c)
        print("Fingerprint", dataset_fingerprint(data))
        count_K = Counter()
        for sender_in, labels, recv_in in data.data:
            K, N, target, nec_features, x = labels
            count_K[K[0]] += 1
            #  print(sender_in[0], nec_features)
            assert(np.allclose(sender_in[0], recv_in[target]))
            assert(not data.ood or not data.is_ood(nec_features))
        print(count_K)


def loaders_from_dataset(dataset, config_data, train_bs, valid_bs,
        shuffle_train=True, num_workers=1):
    N = len(dataset)
    train_size = int(N * (3/5))
    val_size = int(N * (1/5))
    test_size = N - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config_data.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=train_bs,
            shuffle=shuffle_train, num_workers=num_workers, 
            collate_fn=dataset.collater,
            drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=valid_bs,
            shuffle=False, num_workers=num_workers,
            collate_fn=dataset.collater,
            drop_last=False,
    )
    test_loader = DataLoader(test_ds, batch_size=valid_bs,
            shuffle=False, num_workers=num_workers,
            collate_fn=dataset.collater,
            drop_last=False,
    )
    return train_loader, val_loader, test_loader

def init_simple_data(data_cfg, random_seed, batch_size, validation_batch_size,
        shuffle_train, num_workers=1):
    all_data = SimpleData(data_cfg)
    train_loader, valid_loader, test_loader = loaders_from_dataset(
        all_data, data_cfg, batch_size, validation_batch_size,
        shuffle_train, num_workers)
    return all_data, train_loader, valid_loader, test_loader

def get_necessary_features(sender_input):
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
    # except for i=0: we set M[0, j] = 0 for all j.
    necessary = []	
    for i, M in enumerate(diff_to_target):  # iterate over batch
        potential_necessary = [(n_features, tuple(range(n_features)))]
        for comb in powerset(range(0, n_features)):
            # try all combinations of features and test if they're enough
            if len(comb) == 0:
                continue
            comb_tensor = torch.tensor(comb, device=sender_input.device)
            # select the combination of features among distractors only
            D = M[1:n_objs[i].item()].index_select(1, comb_tensor)
            d = torch.all(torch.any(D, 1), 0)
            if d.item():
                potential_necessary.append((len(comb), comb))
        necessary.append(min(potential_necessary, key=lambda x: x[0]))
    #  diff_to_target = diff_to_target.float().sum(-1)  # sum over features
    #  max_, idx = torch.max(diff_to_target[:, 1:], 1)
    #  return max_
    n_necessary = [len_ for len_, _ in necessary]
    necessary_combinations = [comb for _, comb in necessary]
    return torch.tensor(n_necessary).float(), necessary_combinations

def dataset_fingerprint(dataset):
    """ To quickly check that the dataset is regenerated correctly
    (reproducibility).
    """
    first_input = dataset[0][0]
    s = "".join([str(e.item()) for e in first_input.sum(1)])
    s += "".join([str(e.item()) for e in first_input.sum(0)])
    return s

