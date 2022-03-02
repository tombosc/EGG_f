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


class Data(Dataset):
    """ Independent features, independent samples.
    
    The dataset is a list of tuples:
    * sender_input: a matrix where each row is a one hot vector. 
    * label: the index of the first row of sender_input in receiver_input 
    * receiver_input: sender_input with shuffled rows.
    To summarize: sender sees sender_input, has to encode some info about its
    first row *relative to* its other rows, receiver sees a shuffled version of
    receiver_input and has to find label. """

    @dataclass
    class Config(Serializable):
        seed: int = 0
        n_examples: int = 1024
        min_distractors: int = 1
        max_distractors: int = 4
        max_value: int = 10
        n_features: int = 5
 
    def __init__(self, config, necessary_features=False):
        print("Init DAATA")
        # at the time of writing, torch is NOT used for random sampling.
        # but to be future-proof, set the global torch seed:
        c = config
        self.n_features = c.n_features
        self.max_value = c.max_value
        self.min_distractors = c.min_distractors
        self.max_distractors = c.max_distractors
        self.frame = []
        self.necessary_features = necessary_features
        self.rng = np.random.default_rng(c.seed)

        #  self.all_possible_datapoints = np.asarray(list(product(range(1, c.max_value+1),
        #          repeat=c.n_features)))
        #  n_points = self.all_possible_datapoints.shape[0]
        #  print("n_points", n_points)

        #  def generate_example():
        #      #  n_distractors = self.rng.integers(low=c.min_distractors, high=c.max_distractors)
        #      n_distractors = min(self.rng.geometric(p=0.05),
        #              self.max_distractors)
        #      choice = self.rng.choice(n_points, size=n_distractors+1, replace=False)
        #      features = self.all_possible_datapoints[choice]
        #      return features, n_distractors


        for i in range(c.n_examples):
            sender_input, n_distractors = self.generate_example()
            # receiver needs shuffled inputs
            permut = self.rng.permutation(np.arange(0, n_distractors+1))
            label = np.argwhere(permut == 0)[0]
            receiver_input = sender_input[permut]
            if necessary_features:
                sender_input = torch.Tensor(sender_input).long()
                n = get_necessary_features(sender_input.unsqueeze(0))[0]
                self.frame.append((
                    sender_input,
                    torch.Tensor(label).long(),
                    torch.Tensor(receiver_input).long(),
                    n.long(),
                ))
            else:
                self.frame.append((
                    torch.Tensor(sender_input).long(),
                    torch.Tensor(label).long(),
                    torch.Tensor(receiver_input).long(),
                ))

    def generate_example(self):
        n_necessary_features = self.rng.integers(1, self.n_features+1)
        min_distractors = max(self.min_distractors, n_necessary_features)
        max_distractors = min(self.max_distractors,
                              n_necessary_features*(self.max_value)-1)
        if min_distractors == max_distractors:
            n_distractors = max_distractors
        else:
            n_distractors = self.rng.integers(min_distractors, max_distractors)
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
        #  print("Features to change", necessary_features)
        for i in range(1, n_distractors+1):
            feature_to_change = necessary_features[(i-1) %
                    len(necessary_features)]
            features[i] = self.generate_from(
                features[0],
                features[0:i],
                [feature_to_change],
            )
        torched_features = torch.tensor(features).unsqueeze(0)
        assert(get_necessary_features(torched_features)[0].item() == n_necessary_features)
        # Problem: this process makes it possible to perform better than random
        # without transmitting any message! That's because the target has
        # something in common with ALL the others...
        # Potential solution: reshuffle. 
        # n_distractors lose its meaning, and we will generate easier examples.
        # But at least, messages are required to perform better than random.
        #  permut = self.rng.permutation(np.arange(0, n_distractors+1))
        #  features = features[permut]
        return features, n_distractors

    def generate_from(self, vector, set_vectors, changing_features):
        """ Sample a variant from vector where features #changing_features can
        be modified and different from all vectors in set_vectors.
        """
        new_v = vector.copy()
        def valid_vector(candidate):
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

    def get_n_features(self):
        return self.n_features

    def n_combinations(self):
        # example: 3 features, 4 possible values: 4**3
        return self.max_value ** self.n_features

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

    @staticmethod
    def collater(list_tensors):
        inputs = [e[0] for e in list_tensors]
        tgt_index = torch.cat([e[1] for e in list_tensors])
        outputs = [e[2] for e in list_tensors]
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=0)
        necessary_features = len(list_tensors[0]) == 4
        if necessary_features:
            n_necessary_features = torch.cat([e[3] for e in list_tensors])
            return (padded_inputs, tgt_index, padded_outputs, n_necessary_features)
        else:
            return (padded_inputs, tgt_index, padded_outputs)


def generate_quasi_diagonal(rng, n, alpha=0.2):
    """ Generate some sort of "noisy diagonal": diagonal terms are often high
    while off-diagonals are low, and the sum over columns is 1.
    """
    K = rng.dirichlet(alpha=(alpha,)*n, size=n)
    # we're going to align the highest coefficients in the diagonal:
    for i in range(n):
        max_index = K[i].argmax()
        copy = K[i].copy()
        for j in range(n):
            K[i, (i+j) % n] = copy[(max_index + j) % n]
    return K

def generate_sparse_normalized(rng, n, non_null_proba, dirichlet_alpha):
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        while mat[i].sum() == 0:
            bin_ = rng.choice(2, p=[1-non_null_proba,non_null_proba], size=n)
            mat[i] = np.asarray(bin_)
        n_non_null = int(mat[i].sum())
        coeffs = rng.dirichlet((dirichlet_alpha,) * n_non_null)  # sum to 1
        j = 0
        for k in range(n_non_null):
            while mat[i, j] == 0:
                j += 1
            mat[i, j] = coeffs[k]
            j += 1
    return mat

def sample_from_conditionals(rng, conditionals, x_1):
    out = np.zeros((len(conditionals) + 1,), dtype=int)
    out[0] = x_1
    for i, c in enumerate(conditionals):
        x = rng.choice(len(c[x_1]), p=c[x_1])
        out[i+1] = x
    return out

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
        n_features: int = 4
        disentangled: bool = True
        max_distractors: int = 3  # hardcoded, depends on disentangled...
    
    def __init__(self, config):
        c = config
        self.data = []
        rng = np.random.default_rng(c.seed)
        self.n_features = c.n_features
        self.max_value = c.max_value
        # N_distractors: N
        # N_properties needed to distinguish = K
        # p(N) is computed so that P(K) = Σ P(K|N) P(N) is uniform
        
        if c.disentangled:
            assert(c.max_distractors == 3)
            n_distractors = rng.choice(c.max_distractors, p=[1/2, 0, 1/2], size=c.n_examples) + 1
            # in the case where K=2, one chance over 2, in the case where K=4,
            # one over 4 (since equal proba of needing 2 attributes and 1
            # attributes
        else:
            assert(c.max_distractors == 5)
            # in the case where K=4, 1 chances over 2 to get it randomly (since only 2 cases
            # where we need 2 attributes), in the case where K=6, 2 chances
            # over 4
            assert(c.n_features >= 5)
            n_distractors = rng.choice(c.max_distractors, p=[0, 0, 1/2, 0, 1/2], size=c.n_examples) + 1
        zero = np.zeros(c.n_features, dtype=int)
        dict_K = {  # maps (N+1, i) to how many features are needed
            (2, 0): 1, (2, 1): 1,
            (3, 0): 1, (3, 1): 2, (3, 2): 1,
            (4, 0): 1, (4, 1): 2, (4, 2): 2, (4,3): 1,
            (5, 0): 1, (5, 1): 2, (5, 2): 2, (5,3): 2, (5,4): 1,
            (6, 0): 1, (6, 1): 2, (6, 2): 2, (6,3): 2, (6,4): 2, (6,5): 1,
        }
        for N in n_distractors:
            # sample an object uniformly
            x_1 = rng.choice(c.max_value, size=c.n_features) + 1
            prev = x_1
            # to sample distractors, simply modify a random feature 
            distractors = []
            modified_features = []
            for e in range(N):
                distractor = prev.copy()
                possible_features = range_except(c.n_features,
                        modified_features, start=0)
                i_feat = rng.choice(possible_features)
                modified_features.append(i_feat)
                #  print(f"Sel feature = {i_feat} with val {prev[i_feat]}")
                possible_values = range_except(c.max_value, [prev[i_feat]],
                        start=1)
                distractor[i_feat] = rng.choice(possible_values)
                #  print(f"Chosen val = {distractor[i_feat]}")
                distractors.append(distractor)
                prev = distractor
            objects = [x_1] + distractors
            # add padding
            objects += [zero,]*(c.max_distractors - N)
            objects = np.asarray(objects)
            #  print(objects)
            # select target randomly
            if c.disentangled:
                i_target = rng.choice(N+1)
                K = dict_K[(N+1, i_target)]
            else:
                i_target = rng.choice(range(1, N))
                K = dict_K[(N+1, i_target)]
                assert(K > 1)
                # if we do not have disentangled examples, K should be > 1

            #  target = objects[i_target]
            #  A = (target != objects[:N+1]).sum(0)
            #  import pdb; pdb.set_trace()
            sender_input = np.vstack((
                objects[i_target][np.newaxis, :], 
                objects[:i_target],
                objects[i_target+1:]
            ))
            assert(np.allclose(sender_input[0], objects[i_target]))
            K2, necessary_features = get_necessary_features(torch.tensor(sender_input).unsqueeze(0))
            necessary_features = necessary_features[0]  # since we don't pass a batch
            if K2 == 1:
                # padding
                necessary_features = np.concatenate((necessary_features, np.asarray([-1])))
            # verify that the computation of necessary feature yields at least
            # the same number of necessary features as was planned
            assert(K == K2.item())
            labels = (
                np.asarray([K]), np.asarray([N]),
                np.asarray([i_target]),
                necessary_features,
                objects,
            )
            receiver_input = objects
            self.data.append((
                sender_input,
                labels,
                receiver_input,
            ))
            #  print(self.data[-1])

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
        #  padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        #  padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=0)
        return ((p_sender_i, nec_features),
            (K, N, i_target, nec_features),
            p_receiver_i,
        )

class TesterLoss(unittest.TestCase):
    def test_simple_data(self):
        c = SimpleData.Settings(n_examples=4000, max_value=10)
        data = SimpleData(c)
        count_K = Counter()
        for sender_in, labels, recv_in in data.data:
            K, N, target, mod_features, x = labels
            count_K[K[0]] += 1
        print(count_K)

        import pdb; pdb.set_trace()

class DependentData(Dataset):
    @dataclass
    class Settings(Serializable):
        seed: int = 0
        n_examples: int = 1024*5
        min_distractors: int = 1
        max_distractors: int = 15
        max_value: int = 5
        n_features: int = 5
        gen_patience: int = 3

    
    def __init__(self, config, necessary_features=False):
        # at the time of writing, torch is NOT used for random sampling.
        # but to be future-proof, set the global torch seed:
        #  set_torch_seed(config.seed)
        c = config
        self.min_distractors = c.min_distractors
        self.max_distractors = c.max_distractors
        self.frame = []
        self.necessary_features = necessary_features
        rng = np.random.default_rng(c.seed)
        self.n_features = c.n_features
        self.max_value = c.max_value
        # an example consists of n+1 vectors: 1 target and n distractors.
        # each vector is sampled from the same distribution.
        # however, vectors in an example are correlated (drawn using markov
        # chains) AND features are not independently drawn either.
        # transition matrix for sampling correlated x_1
        K_1 = generate_quasi_diagonal(rng, c.max_value, alpha=0.2)
        init_P_1 = rng.dirichlet(alpha=(0.8,)*c.max_value)
        # define p(x_2|x_1), p(x_3|x_2), and so forth
        conditionals = []
        for _ in range(c.n_features - 1):
            non_null_p = rng.uniform(low=0.2, high=0.8)
            alpha = rng.uniform(
                low=non_null_p,  # if very sparse, we don't want very peaked
                # coefficients either.
                high=1,
            )
            cond = generate_sparse_normalized(
                rng, c.max_value, non_null_proba=non_null_p,
                dirichlet_alpha=0.5,
            )
            conditionals.append(cond)
        for i in range(c.n_examples):
            n_objects = 1 + self.min_distractors + scipy.stats.betabinom.rvs(
                self.max_distractors - self.min_distractors,
                0.5,
                0.3,
                random_state=i, # can't pass the numpy rng... and can't get the
                # rng's seed.
            )
            #  n_objects = 1 + rng.integers(low=self.min_distractors,
            #                         high=self.max_distractors + 1)
            x_1 = rng.choice(c.max_value, p=init_P_1)
            objects = np.zeros((n_objects, c.n_features), dtype=int)
            objects[0] = sample_from_conditionals(rng, conditionals, x_1)
            i = 1
            level = c.n_features - 2 # resample only last feature
            while i < n_objects:
                gen_patience = c.gen_patience
                while level >= 0:
                    # strategy: start by re-sampling only the last feature
                    # if it yields an already exisiting vector in objects, 
                    # go up a level and resample the before last feature, etc.
                    last_object = objects[i-1]
                    incomplete_x = last_object[:level]
                    if level == 0:
                        x_1 = rng.choice(c.max_value, p=K_1[last_object[0]])
                        v = sample_from_conditionals(rng, conditionals, x_1)
                    else:
                        u = sample_from_conditionals(rng, conditionals[level:],
                                                     incomplete_x[level-1])
                        v = np.concatenate((incomplete_x, u))
                    if not np.all(np.any(v != objects[:i], axis=1), 0):
                        if gen_patience > 0:
                            gen_patience -= 1
                            continue
                        if level > 0:
                            gen_patience = c.gen_patience
                            level -= 1
                        continue
                    else:
                        objects[i] = v
                        i += 1
                        break
            sender_input = objects + 1
            permut = rng.permutation(np.arange(0, n_objects))
            sender_input = sender_input[permut]
            permut = rng.permutation(np.arange(0, n_objects))
            label = np.argwhere(permut == 0)[0]
            receiver_input = sender_input[permut]
            if necessary_features:
                sender_input = torch.Tensor(sender_input).long()
                n, _ = get_necessary_features(sender_input.unsqueeze(0))
                self.frame.append((
                    sender_input,
                    torch.Tensor(label).long(),
                    torch.Tensor(receiver_input).long(),
                    n.long(),
                ))
            else:
                self.frame.append((
                    torch.Tensor(sender_input).long(),
                    torch.Tensor(label).long(),
                    torch.Tensor(receiver_input).long(),
                ))

    def get_n_features(self):
        return self.n_features

    def n_combinations(self):
        # example: 3 features, 4 possible values: 4**3
        # TODO that is NOT correct. 
        return self.max_value ** self.n_features

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

    def n_distractors(self, sender_input):
        return torch.all(sender_input != 0, dim=-1).float().sum(1) - 1



def loaders_from_dataset(dataset, config_data, train_bs, valid_bs):
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
            shuffle=True, num_workers=1, collate_fn=dataset.collater,
            drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=valid_bs,
            shuffle=False, num_workers=1, collate_fn=dataset.collater,
            drop_last=True,
    )
    test_loader = DataLoader(test_ds, batch_size=valid_bs,
            shuffle=False, num_workers=1, collate_fn=dataset.collater,
            drop_last=True,
    )
    return train_loader, val_loader, test_loader

def init_dependent_data(data_cfg, random_seed, batch_size,
        validation_batch_size):
    all_data = DependentData(data_cfg)
    train_loader, valid_loader, test_loader = loaders_from_dataset(
        all_data, data_cfg, batch_size, validation_batch_size)
    return all_data, train_loader, valid_loader, test_loader


def init_simple_data(data_cfg, random_seed, batch_size, validation_batch_size):
    all_data = SimpleData(data_cfg)
    train_loader, valid_loader, test_loader = loaders_from_dataset(
        all_data, data_cfg, batch_size, validation_batch_size)
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

