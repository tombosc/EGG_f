# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def strvec2intvec(v):
    return list(map(int, v))


def to_one_hot(l, n_values):
    """ Returns a torch one-hot vector given a python list l.
    """
    v = np.zeros((len(l), n_values), dtype=np.int8)
    v[range(len(l)), l] = 1
    return torch.Tensor(v)


# TODO: copy description, delete??
class Data(Dataset):
    """ The dataset is a list of tuples:
    * sender_input: a matrix where each row is a one hot vector. 
    * label: the index of the first row of sender_input in receiver_input 
    * receiver_input: sender_input with shuffled rows.
    To summarize: sender sees sender_input, has to encode some info about its
    first row *relative to* its other rows, receiver sees a shuffled version of
    receiver_input and has to find label. """

    def __init__(self, path, one_hot):
        """ The file in path looks like that: 
        n_feature max_value
        a b c d . a' b' c' d' . ... . n0
        e f g h . e' f' g' h' . ... . n1
        where a, a', b, b' ... are ints between 0 and max_value
        and len([a, b, c, d]) = n_feature (here = 4)
        """
        #  print("reading from", path)
        with open(path, 'r') as f:
            # brutally read everything in ram and parse
            self.parse(f.readlines(), one_hot)

    @classmethod
    def from_str(cls, str_, one_hot):
        with tempfile.NamedTemporaryFile() as f:
            f.write(str_)
            f.seek(0) 
            return Data(f.name, one_hot)

    def parse(self, lines, one_hot):
        self.frame = []
        self.n_features, self.max_value = [int(e) for e in lines[0].split(' ')]
        for line in lines[1:]:
            row_info = line.split('.')
            target_index = int(row_info[-1])
            vectors = [strvec2intvec(v.strip().split(' ')) for v in row_info[:-1]]
            # receiver input is ordered according to the file's order
            # sender input will see the target first, and a shuffling of
            # the order in the file, i.e.:
            # sender_input[0] == receiver_input[label]
            vectors_minus_targets = [v for i, v in enumerate(vectors) if i != target_index]
            np.random.shuffle(vectors_minus_targets)
            shuf_vectors = [vectors[target_index]] + vectors_minus_targets
            if one_hot:
                one_hots = [to_one_hot(v, N) for v in vectors]
                shuf_one_hots = [to_one_hot(v, N) for v in shuf_vectors]
                self.frame.append((
                    shuf_one_hots,
                    target_index,
                    one_hots,
                ))
            else:
                self.frame.append((
                    torch.Tensor(shuf_vectors),
                    target_index,
                    torch.Tensor(vectors),
                ))

        if one_hot:
            # UNTESTED
            raise NotImplementedError()
            n = [len(e[0]) for e in self.frame]
            inputs = [torch.stack(e[0]) for e in self.frame]
        else:
            inputs = [e[0] for e in self.frame]
            n = [len(e[0]) for e in self.frame]
            outputs = [e[2] for e in self.frame]
        print(len(inputs))
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=-1)
        padded_inputs = list(zip(padded_inputs, n))
        padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=-1)
        tgt_index = [e[1] for e in self.frame]
        self.frame = list(zip(padded_inputs, tgt_index, padded_outputs))

    def get_n_features(self):
        #  return self.frame[0][0][0].size(0)
        return self.n_features

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

class GeneratedData(Dataset):
    def __init__(self, N, max_value, min_distractors, max_distractors,
            n_features, seed):
        assert(min_distractors > 0)
        self.n_features = n_features
        self.max_value = max_value
        self.frame = []
        rng = np.random.default_rng(seed)

        def generate_example():
            # TODO is it max_value, or max_value+1 in embedding?
            # TODO assert that there can't be 0 distractors, it seems to happen
            n_distractors = rng.integers(low=min_distractors, high=max_distractors)
            label = rng.choice(n_distractors+1) 
            features = rng.integers(low=1, high=max_value+1,
                                    size=(n_distractors+1,
                                              n_features))
            return (features, label)

        for i in range(N):
            example, label = generate_example()
            target = example[label]
            distractors = np.concatenate((example[:label],
                                          example[label+1:]))
            rng.shuffle(distractors)
            x = np.concatenate((target.reshape((1, -1)), distractors))
            self.frame.append((
                torch.Tensor(x).long(),
                torch.Tensor([label]).long(),
                torch.Tensor(example).long(),
            ))

    def get_n_features(self):
        return self.n_features

    def n_combinations(self):
        # example: 3 features, 4 possible values: 4**3
        return self.max_value ** self.n_features

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]


