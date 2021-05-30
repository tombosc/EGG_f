import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data
import csv
from collections import defaultdict, namedtuple, Counter
import itertools 
from torch._utils import _accumulate
from dataclasses import dataclass
from simple_parsing.helpers import Serializable


VerbTokenArg = namedtuple('VerbTokenArg',
    ['split', 'sentence_id', 'pred_token', 'arg', 'roleset',
    'gram_func', 'properties'])
VerbToken = namedtuple('VerbToken',
    ['split', 'sentence_id', 'pred_token', 'roleset',
    'gram_funcs', 'properties'])

class Data(data.Dataset):
    """ Proto-roles.
    """
    @dataclass
    class Settings(Serializable):
        train_ratio: float = 0.7
        valid_ratio: float = 0.1
        dataset_seed: int = 0
        shuffle_roles: bool = False
        augment: str = 'basic'  # none or basic

    @staticmethod
    def compactify(L, set_properties, n_max_args):
        """ In L, every element contains a single property value for a certain
        verb token and a certain argument of that verb token. This function
        turns it into a list where there is only one element per verb token.
        We store all properties in a 2D array where the rows indicate the arg
        number (which has a meaning in terms of classical roles).
        """
        compacted = {}
        ordered_properties = sorted(list(set_properties))
        numbered_prop = {p: i for i, p in enumerate(ordered_properties)}
        #  print("Numbered prop", numbered_prop)
        n_prop = len(set_properties)
        arg_pos_per_key = defaultdict(set)
        for l in L:
            arg = int(l['Arg'])
            if arg >= n_max_args:
                continue
            # why do we need such a long index?
            # - there are several sentences
            # - in each sentence, there might be several verb tokens 
            # - for each verb token, there are often several arguments
            # - for each argument, there might be several Arg.Pos? This doesn't
            #   make sense to me. If we don't, we have 9704 instead of 9738 ex
            #   I remove the duplicates, see below.
            key = (l['Sentence.ID'], l['Pred.Token'], arg)#, l['Arg.Pos'])
            # TODO do that above, will save time
            entry = compacted.get(key, None)
            arg_pos_per_key[key].add(l['Arg.Pos'])
            if not entry:
                entry = VerbTokenArg(
                    split=l['Split'], sentence_id=l['Sentence.ID'],
                    pred_token=l['Pred.Token'], arg=arg,
                    roleset=l['Roleset'], gram_func=l['Gram.Func'],
                    properties=np.zeros(n_prop, dtype=int),
                )
                compacted[key] = entry
            applicable = eval(l['Applicable'])
            if not applicable:
                # nothing to do, b/c array is initialized to N/A value = 0
                continue
            # add the property
            k = numbered_prop[l['Property']]
            response = l['Response']
            entry.properties[k] = response
        # I don't understand how this can be, so I remove these for now
        # it's only a small minority (we get 9670 lines instead of 9738)
        problematic_keys = set([k for k, v in arg_pos_per_key.items() if len(v) > 1])
        compacted = {k: v for k, v in compacted.items() if k not in problematic_keys}
        for e in compacted.values():
            for i in range(e.properties.shape[0]):
                # diminish # of categories: since ordinal data is treated as
                # categorical data and 2 and 4 ratings are very rare, it is
                # better to convert them.
                if e.properties[i] == 2:
                    e.properties[i] = 1
                if e.properties[i] == 4:
                    e.properties[i] = 5
        print("total sentence, tokens and arguments:", len(compacted))
        # compacted is indexed by sentence_id AND argument number
        # we now want to build a new representation only indexed by sentence_id
        # indeed, a datapoint is going to consist in an event with several
        # arguments
        tokens = {}
        idx_gram_func = {'subj': 0, 'obj': 1, 'other': 2}
        count_args = Counter()
        for (sent_id, pred_token, arg), vta in compacted.items():
            if arg > n_max_args:
                continue
            key = (sent_id, pred_token)
            verb_token = tokens.get(key, None)
            count_args[arg] += 1
            if not verb_token:
                # instantiate array
                properties = np.zeros((n_max_args, n_prop), dtype=int)
                gram_funcs = np.ones(n_max_args, dtype=int) * -1
                verb_token = VerbToken(
                    split=vta.split, sentence_id=vta.sentence_id, 
                    pred_token=vta.pred_token, roleset=vta.roleset, 
                    gram_funcs=gram_funcs, properties=properties,
                )
                tokens[key] = verb_token
            else:
                properties = verb_token.properties
            properties[arg, :] = vta.properties
            #  if np.all(vta.properties == 0):
            #      print("Error?")
            #      print(vta)
            verb_token.gram_funcs[arg] = idx_gram_func[vta.gram_func]
        print("total sentences and verb tokens:", len(tokens))
        print("args count", count_args)
        return tokens, ordered_properties


    def __init__(self, seed=0, n_max_args=3, 
                 fn="protoroles_eng_pb/protoroles_eng_pb_08302015.tsv",
                 augment="basic", shuffle_roles=False,
        ):
        """
        For values of augment, please see prepare_inputs
        shuffle_roles: if True, for each verb, roles are going to be shuffled.
        """
        rng = np.random.default_rng(seed)
        lines = []
        set_properties = set()
        with open(fn) as f:
            csv_reader = csv.DictReader(f, delimiter='\t')
            # group each properties in a list
            for line in csv_reader:
                #  key = (line['Sentence.ID'], line['Arg'])
                #  verb_tokens_raw[key].append(line)
                set_properties.add(line['Property'])
                lines.append(line)
        # then, turn each one of them into a list of integers
        verb_tokens, self.ordered_properties = \
                self.compactify(lines, set_properties, n_max_args)
        # we can now group all verb tokens to get the distribution over types
        def present_roles(vt):
            #  n_total = vt.properties.shape[0]
            not_0 = tuple((vt.properties != 0).any(1))
            return not_0

        def n_roles(vt):
            #  n_total = vt.properties.shape[0]
            not_0 = (vt.properties != 0).any(1).sum()
            return not_0
        counts = Counter([present_roles(vt) for vt in verb_tokens.values()])
        keys_all_0 = [k for k, vt in verb_tokens.items() if n_roles(vt) == 0]
        print("removing {} examples which have no roles (n_max_args={})".format(
            len(keys_all_0), n_max_args))
        print(counts)
        verb_tokens = {k: v for k, v in verb_tokens.items() if k not in keys_all_0}
        # we need to encode the roleset (the verb + an ID for polysemy) as an integer
        #  role_counters = Counter([vt.roleset for vt in verb_tokens.values()])
        rolesets = sorted(list(set([vt.roleset for vt in verb_tokens.values()])))
        self.idx_roleset = {role: i for i, role in enumerate(rolesets)}
        examples = []
        # we have to iterate over the sorted keys to stay deterministic...
        sorted_keys = sorted(verb_tokens.keys())
        if shuffle_roles:
            roleset_permutations = [rng.permutation(n_max_args) for _ in self.idx_roleset]
        for k in sorted_keys:
            v = verb_tokens[k]
            i = self.idx_roleset[v.roleset]
            if shuffle_roles:
                perm = roleset_permutations[i]
                properties = v.properties[perm]
                gram_funcs = v.gram_funcs[perm]
            else:
                properties = v.properties
                gram_funcs = v.gram_funcs
            examples.append((properties, gram_funcs, i))
        # 1. augment the dataset to hide some slots
        self.examples = []
        # we're doing data augmentation. we want to make sure that augmented
        # examples from a single original example are all gathered in the same
        # split. that's what tied_ids is for.
        self.tied_ids = []  
        for orig_example in examples:
            new_examples = self.prepare_inputs(*orig_example, augment=augment)
            # for each verb token in the original dataset, we want to keep the
            # "augmented" ones (with partial inputs) in the same split, so we
            # need to store the ids
            cur_id = len(self.examples)
            bunch_of_ids = range(cur_id, cur_id + len(new_examples))
            self.tied_ids.append(list(bunch_of_ids))
            self.examples.extend(new_examples)

    @staticmethod
    def prepare_inputs(properties, gram_funcs, id_roleset, augment):
        """ Given an example consisting of properties and their grammatical
        functions (-1 if unused, 0 if subj, 1 if obj, 2 if other), this
        function returns a list of inputs (sender_inputs, labels, receiver_inputs)
        where all the possible combinations of non -1 functions are hidden 
        (set to 0).
        """
        # first, we need to preprocess the properties a bit
        # in the input, 0: N/A, 1, 3, 5: values of Likert scale
        # since I have removed values 2 and 4 which were very rare,
        # I can now code 3 and 5 as follows:
        properties[properties == 3] = 2
        properties[properties == 5] = 3

        # first, add the original example (see comments below)
        def package_input(to_send, hide_all_receiver=False, hide_roleset=False): 
            if hide_all_receiver:
                incomplete_properties = np.zeros_like(properties)
            else:
                # the receiver sees the incomplete properties
                incomplete_properties = properties.copy()
                incomplete_properties *= (1 - to_send[1:, np.newaxis])
            if hide_roleset:
                rcv_roleset = 0
            else: 
                rcv_roleset = id_roleset + 1
            receiver_input = (rcv_roleset, incomplete_properties)
            labels = (id_roleset, properties, gram_funcs)
            # the sender has the original input, + an indication of what to
            # transmit to the receiver
            sender_input = (id_roleset+1, properties, to_send)
            return (sender_input, labels, receiver_input)
        
        filled_roles = [i for i, f in enumerate(gram_funcs) if f != -1]

        if augment == "none":
            original_roles = np.zeros(1 + gram_funcs.shape[0], dtype=int)
            original_roles[1:] = (np.asarray(gram_funcs) >= 0)
            original_example = package_input(original_roles)
            return [original_example]
        
        if augment == "basic":
            combinations_to_send = []
            for j in range(1, len(filled_roles) + 1):
                iter_ = itertools.combinations(filled_roles, j)
                combinations_to_send += list(iter_)

            new_examples = []
            for combination in combinations_to_send:
                combination_array = np.zeros(1 + gram_funcs.shape[0], dtype=int)
                combination_array[list([c+1 for c in combination])] = 1
                # have to send part of the information
                new_examples.append(package_input(combination_array))
            return new_examples

    #  def analyse_role_distributions(self, selection, n_max_args):
    #      n_roles = np.zeros(n_max_args)
    #      prop_arrays = []
    #      for e in selection:
    #          n_roles += (e.gram_funcs != -1)
    #          prop = e.properties.copy()
    #          prop[prop == 0] = np.nan
    #          prop_arrays.append(prop)
    #      mean = np.nanmean(np.stack(prop_arrays), axis=0)
    #      print(n_roles)
    #      print(mean)

    def random_split(self, ratios, generator):
        r""" A modification of random_split so that each augmented/noisy
            example stays in the same split with the original one.
        """
        assert(sum(ratios) == 1.0)
        # importantly, we do not create permutations on self.examples!
        # we create permutations on the lists of ids that essentially encode
        # the same "situation"!
        n_ids = len(self.tied_ids)
        group_indices = torch.randperm(n_ids, generator=generator).tolist()
        # let's deal with non-integerness that way:
        lengths = [int(r*n_ids) for r in ratios[:-1]]
        lengths.append(n_ids - sum(lengths))
        subsets = []
        for offset, length in zip(_accumulate(lengths), lengths):
            groups = group_indices[offset - length : offset]
            indices = [e for g in groups for e in self.tied_ids[g]]
            subsets.append(data.Subset(self, indices))
        return subsets

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # data consists in verb type and property matrix and verb type
        return self.examples[i]
