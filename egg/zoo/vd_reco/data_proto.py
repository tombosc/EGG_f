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
from .callbacks import entropy_list


VerbTokenArg = namedtuple('VerbTokenArg',
    ['split', 'sentence_id', 'pred_token', 'arg', 'roleset',
    'gram_func', 'properties'])

VerbToken = namedtuple('VerbToken',
    ['split', 'sentence_id', 'pred_token', 'roleset',
    'gram_funcs', 'properties', 'classical_roles'])


def counter_to_set_and_probas(c):
    list_values_counts = list(c.items())
    values, counts = list(zip(*list_values_counts))
    counts = np.asarray(counts)
    counts = counts / counts.sum()
    return values, counts


class Data(data.Dataset):
    """ Proto-roles.
    """
    @dataclass
    class Settings(Serializable):
        dataset: str = 'proto'  
        train_ratio: float = 0.7
        valid_ratio: float = 0.1
        dataset_seed: int = 2147483647
        shuffle_roles: bool = False
        augment: str = 'basic'  # none or basic
        hide_to_send: bool = False
        n_thematic_roles: int = 3

    @staticmethod
    def aggregate(L, set_properties, n_thematic_roles):
        """ Every element of L contains 1 property value annotated for 1 
        verb token and a 1 argument of that verb token. This function
        aggregates L into a list where there is only 1 element per verb token:
        it stores all properties in a 2D array where the rows index the arg
        number (which has a meaning in terms of classical roles) and the
        columns index the property.
        """
        compacted = {}
        ordered_properties = sorted(list(set_properties))
        numbered_prop = {p: i for i, p in enumerate(ordered_properties)}
        print("Numbered prop", numbered_prop)
        n_prop = len(set_properties)
        arg_pos_per_key = defaultdict(set)
        for l in L:
            arg = int(l['Arg'])
            if arg >= n_thematic_roles:
                continue
            key = (l['Sentence.ID'], l['Pred.Token'], arg)#, l['Arg.Pos'])
            # The key is there to aggregate different elements of L into a
            # single VerbTokenArg in a new list. Why such a key?
            # - there are several sentences
            # - in each sentence, there might be several verb tokens 
            # - for each verb token, there are often several arguments
            # - for each argument, there might be several Arg.Pos? I don't
            #   understand that, but if we omit, we get 9704 instead of 9738 
            #   elements. I remove the duplicates, see arg_pos_per_key.
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
                # skip assigment of prop, b/c array is initialized to NA (0)
                continue
            # assign property
            k = numbered_prop[l['Property']]
            entry.properties[k] = l['Response']
        # I don't understand how this can be, so I remove these for now
        # it's only a small minority (we get 9670 lines instead of 9738)
        problematic_keys = set([k for k, v in arg_pos_per_key.items() if len(v) > 1])
        compacted = {k: v for k, v in compacted.items() if k not in problematic_keys}
        for e in compacted.values():
            for i in range(e.properties.shape[0]):
                # diminish # of categories: since ordinal data is treated as
                # categorical data and 2 and 4 ratings are rare, we might
                # as well convert them
                if e.properties[i] == 2:
                    e.properties[i] = 1
                if e.properties[i] == 4:
                    e.properties[i] = 5
        print("total sentence, tokens and arguments:", len(compacted))
        # compacted is indexed by sentence_id AND argument number
        # we now want to build a new representation only indexed by sentence_id
        # that is, aggregate all the arguments of the same verb token into a 
        # single data structure.
        # indeed, a datapoint fed to the model is going to consist in an event
        # with several arguments
        tokens = {}
        idx_gram_func = {'subj': 0, 'obj': 1, 'other': 2}
        count_args = Counter()
        for (sent_id, pred_token, arg), vta in compacted.items():
            if arg > n_thematic_roles:
                continue
            key = (sent_id, pred_token)
            verb_token = tokens.get(key, None)
            count_args[arg] += 1
            if not verb_token:
                # instantiate array
                properties = np.zeros((n_thematic_roles, n_prop), dtype=int)
                gram_funcs = np.ones(n_thematic_roles, dtype=int) * -1
                roles = np.ones(n_thematic_roles, dtype=int) * -1
                verb_token = VerbToken(
                    split=vta.split, sentence_id=vta.sentence_id, 
                    pred_token=vta.pred_token, roleset=vta.roleset, 
                    gram_funcs=gram_funcs, properties=properties,
                    classical_roles=roles,
                )
                tokens[key] = verb_token
            else:
                properties = verb_token.properties
            properties[arg, :] = vta.properties
            verb_token.classical_roles[arg] = arg
            verb_token.gram_funcs[arg] = idx_gram_func[vta.gram_func]
        print("total sentences and verb tokens:", len(tokens))
        print("args count", count_args)
        return tokens, ordered_properties

    def __init__(self, seed=0, n_thematic_roles=3, 
                 fn="protoroles_eng_pb/protoroles_eng_pb_08302015.tsv",
                 augment="basic", shuffle_roles=False, hide_to_send=False,
        ):
        """
        For values of augment, please see prepare_inputs
        shuffle_roles: if True, for each verb, roles are going to be shuffled.
        """
        self.n_thematic_roles = n_thematic_roles
        self.hide_to_send = hide_to_send
        self.seed = seed
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
                self.aggregate(lines, set_properties, n_thematic_roles)
        # we can now group all verb tokens to get the distribution over types
        def present_roles(vt):
            # (vt.classical_roles[j] == -1) iff no entity takes the jth role
            return tuple((vt.classical_roles != -1))
        # I got rid of all examples (48 examples) where all entities only 
        # have NA features, but I don't remember why I decided to do that and 
        # forgot about it. Consider it a small bug left for reproducibility...
        def n_non_NA(vt):
            return (vt.properties != 0).any(1).sum()
        keys_all_NA = [k for k, vt in verb_tokens.items() if n_non_NA(vt) == 0]
        print("removing {} examples where all features of all entities are NA (n_thematic_roles={})".format(
            len(keys_all_NA), n_thematic_roles))
        # show repartition of roles
        counts = Counter([present_roles(vt) for vt in verb_tokens.values()])
        print("Role distribution across examples:", counts)
        verb_tokens = {k: v for k, v in verb_tokens.items() if k not in keys_all_NA}
        # we need to encode the roleset (the verb + an ID for polysemy) as an integer
        #  role_counters = Counter([vt.roleset for vt in verb_tokens.values()])
        self.ordered_rolesets = sorted(list(set([vt.roleset for vt in verb_tokens.values()])))
        self.roleset2idx = {role: i for i, role in enumerate(self.ordered_rolesets)}
        examples = []
        # we have to iterate over the sorted keys to stay deterministic...
        sorted_keys = sorted(verb_tokens.keys())
        if shuffle_roles:
            self.roleset_permutations = [rng.permutation(n_thematic_roles) for
                    _ in self.roleset2idx]
        for k in sorted_keys:
            v = verb_tokens[k]
            i = self.roleset2idx[v.roleset]
            if shuffle_roles:
                perm = self.roleset_permutations[i]
                properties = v.properties[perm]
                gram_funcs = v.gram_funcs[perm]
                roles = v.classical_roles[perm]
                example = (properties, gram_funcs, i, perm.astype(float),
                        roles, v.sentence_id)
            else:
                properties = v.properties
                gram_funcs = v.gram_funcs
                roles = v.classical_roles
                example = (properties, gram_funcs, i, None, roles,
                        v.sentence_id)
            examples.append(example)
        # 1. augment the dataset to hide some slots
        self.examples = []
        # we're doing data augmentation. we want to make sure that augmented
        # examples from a single original example are all gathered in the same
        # split. that's what tied_ids is for.
        self.tied_ids = []  
        for orig_example in examples:
            new_examples = self.prepare_inputs(*orig_example, augment=augment,
                    hide_to_send=self.hide_to_send)
            # for each verb token in the original dataset, we want to keep the
            # "augmented" ones (with partial inputs) in the same split, so we
            # need to store the ids
            cur_id = len(self.examples)
            bunch_of_ids = range(cur_id, cur_id + len(new_examples))
            self.tied_ids.append(list(bunch_of_ids))
            self.examples.extend(new_examples)

    @staticmethod
    def prepare_inputs(properties, gram_funcs, id_roleset, permutation,
            thematic_roles, sentence_id, augment, hide_to_send):
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
                # the receiver sees the incomplete properties and th roles
                incomplete_properties = properties.copy()
                incomplete_properties *= (1 - to_send[1:, np.newaxis])
                incomplete_th_roles = thematic_roles.copy()
                incomplete_th_roles[to_send[1:] == 1] = -1
            if hide_roleset:
                rcv_roleset = 0
            else: 
                rcv_roleset = id_roleset + 1
            receiver_input = (rcv_roleset, incomplete_properties,
                    incomplete_th_roles)
            zero = np.zeros(1)
            labels = [id_roleset, properties, gram_funcs, zero,
                    thematic_roles, sentence_id]
            if permutation is not None:
                labels[3] = permutation
            labels = tuple(labels)
            # the sender has the original input, + an indication of what to
            # transmit to the receiver, unless hide_to_send!
            if hide_to_send:
                to_send[:] = 0
            sender_input = (id_roleset+1, properties, thematic_roles, to_send)
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

    def random_split(self, ratios, generator):
        r""" A modification of random_split so that each augmented/noisy
            example stays in the same split with the original one.
        """
        assert(sum(ratios) == 1.0)
        # importantly, we do not create permutations on self.examples directly
        # we create permutations on the lists of ids that encodes
        # the same "situation".
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
        return self.examples[i] + (i,)

    def generate_chimeras(self):
        """ Re-sample a similar dataset where object-object cooccurences and 
        role-object cooccurences are sampled from product of marginals.
        Ignore valid/test set splits: this is only for test purposes
        """
        N = len(self.examples)
        #  thematic_roles_counts = Counter()
        new_examples = []
        relations_counts = Counter()
        entity_counts = Counter()
        rng = np.random.default_rng(self.seed)
        for speaker_in, _, _ in self.examples:
            id_roleset, properties, thematic_roles, to_send = speaker_in
            id_roleset -= 1
            #  thematic_roles_counts[tuple(to_send.tolist())] += 1
            relations_counts[id_roleset] += 1
            for entity_vec, role in zip(properties, thematic_roles):
                if role > 0:
                    entity_counts[tuple(entity_vec.tolist())] += 1
        rel_val, rel_prob = counter_to_set_and_probas(relations_counts) 
        new_relations = rng.choice(rel_val, p=rel_prob, size=N)
        #  role_val, role_prob = counter_to_set_and_probas(thematic_roles_counts)
        #  role_map = {i: v for i, v in enumerate(role_val)}
        #  new_roles = rng.choice(len(role_val), p=role_prob, size=N)
        entities, entity_probs = counter_to_set_and_probas(entity_counts)
        entities_map = {i: np.asarray(v) for i, v in enumerate(entities)}
        new_examples = []
        for i, example in enumerate(self.examples):
            speaker_in, labels, listener_in = example
            gram_funcs = labels[2]
            roles = labels[4]
            #  new_th_roles = role_map[new_roles[i]]
            full_prop = speaker_in[1].copy()
            partial_prop = listener_in[1].copy()
            #  print("full", full_prop)
            #  print("partial", partial_prop)
            for k, role in enumerate(labels[4][1:]):
                if role < 0:
                    continue
                j = rng.choice(len(entities), p=entity_probs)
                new_entity = entities_map[j]
                full_prop[k,:] = new_entity
                to_send_k = speaker_in[3][k+1]
                if to_send_k == 0:
                    partial_prop[k,:] = new_entity
            #  print("full", full_prop)
            #  print("partial", partial_prop)
            speaker_in = (new_relations[i] + 1, full_prop, *speaker_in[2:])
            listener_in = (new_relations[i] + 1, partial_prop, *listener_in[2:])
            labels = (new_relations[i], full_prop, *labels[2:])
            new_examples.append((speaker_in, labels, listener_in))
            if i == 0:
                print(new_examples[0])
        self.examples = new_examples

    def count_arguments(self, loader):
        args = Counter()
        for e in loader:
            inputs_S, labels, inputs_R, id_ = e
            assert(labels[4].size(0) == 1)  # batch size should be 1!
            thematic_roles = labels[4][0]
            properties = labels[1][0]
            for pos, role in enumerate(thematic_roles):
                if role == -1:
                    continue
                args[tuple(properties[pos].tolist())] += 1
        return args
    
    def get_entity_repr(self, prop2idx, properties, thematic_roles):
        """ From a property array and classical roles (-1 if missing, 0, 1, or
        2 else), return an array with entities number of -1 if entity is
        missing, -2 if it is unknown.
        """
        indices = []
        for role, prop in zip(thematic_roles, properties):
            idx = prop2idx.get(tuple(prop.tolist()), -2) if role != -1 else -1
            indices.append(idx)
        return tuple(indices)

    def count_arg_tuples(self, prop2idx, loader):
        arg_tuples = Counter()
        for e in loader:
            inputs_S, labels, inputs_R, id_ = e
            assert(labels[4].size(0) == 1)  # batch size should be 1!
            thematic_roles = labels[4][0]
            properties = labels[1][0]
            entities = self.get_entity_repr(prop2idx, properties, thematic_roles)
            arg_tuples[entities] += 1
        return arg_tuples
 

    def print_filtered_examples(self, arg_index, prop2idx, entities=None,
            roleset=None, use_nltk=True, skip_shown=True):
        """ Find the list of examples with entities and roleset in the train
        set.
        Can set either params to None to disable filtering on these.
        """
        assert(entities is not None or roleset is not None or split is not None)
        if entities is not None and type(entities) == tuple:
            entities_array = np.asarray([list(arg_index[e]) for e in entities])
        elif entities is not None and type(entities) == int:
            entity_array = np.asarray(list(arg_index[entities]))
        elif entities is not None:
            raise NotImplementedError()
        if use_nltk:
            from nltk.corpus import ptb
        already_shown = set()
        for i, e in enumerate(self.examples):
            inputs_S, labels, inputs_R = e
            thematic_roles = labels[4]
            properties = labels[1]
            example_roleset = labels[0]
            if (roleset is not None) and (example_roleset != roleset):
                continue
            if (entities is not None):
                if type(entities) == tuple and (not np.allclose(entities_array, labels[1])):
                    continue
                if type(entities) == int:
                    matching_role = (entity_array == labels[1]).all(1)
                    if not matching_role.any():
                        continue
            if labels[5] in already_shown and skip_shown:
                continue
            filename, sent_id = labels[5].split('_')
            if use_nltk:
                dirname = filename[:2]
                tokens = ptb.sents(fileids=f"WSJ/{dirname}/WSJ_{filename}.MRG")[int(sent_id)]
                role = self.ordered_rolesets[example_roleset]
                E = self.get_entity_repr(prop2idx, properties, thematic_roles)
                print(f"----- verb={role}, entities={E}")
                print(f"Sentence {i}:" + " ".join(tokens))
                already_shown.add(labels[5])

def init_data(data_cfg, run_random_seed, batch_size):
    """ Everything data loading related here.
    """
    data_gen = torch.Generator()
    data_gen.manual_seed(data_cfg.dataset_seed)
    all_data = Data(seed=data_cfg.dataset_seed, augment=data_cfg.augment, 
                     shuffle_roles=data_cfg.shuffle_roles,
                     hide_to_send=data_cfg.hide_to_send,
                     n_thematic_roles=data_cfg.n_thematic_roles, 
                   )
    ratios = (data_cfg.train_ratio, data_cfg.valid_ratio,
              1 - (data_cfg.train_ratio + data_cfg.valid_ratio))
    train_data, valid_data, test_data = all_data.random_split(ratios, data_gen)
    # data ordering determined by same random seed as model params,
    # not as dataset random seed (which determines train/test split)
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(run_random_seed)
    # BatchSampler didn't work, but luckily there's an undocumented generator
    # parameter for DataLoader.
    train_loader = data.DataLoader(train_data, batch_size=batch_size,
            shuffle=True, generator=shuffle_gen)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)
    return all_data, train_loader, valid_loader, test_loader

def dataset_to_chimeras(dataset, batch_size):
    """ Careful! Modify the dataset IN PLACE.
    """
    dataset.generate_chimeras()
    dummy_indices = range(0,1)
    main_data_indices = range(1, len(dataset))
    data_loader = data.DataLoader(
        data.Subset(dataset, main_data_indices),
        batch_size=batch_size,
        shuffle=False, 
    )
    dummy_loader = data.DataLoader(
        data.Subset(dataset, dummy_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    return data_loader, dummy_loader


if __name__ == '__main__':
    # here are a bunch of unrelated things to check on the dataset
    data_cfg = Data.Settings.load('res_proto_1B_adam/c62c94b2c6149580d5a354d984a3287a_I/data.json')
    data, trainDL, validDL, testDL = init_data(data_cfg, 0, 1);
    # first, compute the entropy of the roles on D_1 (the subset of the dataset
    # where there is a single hidden entity):
    count_roles_D1 = Counter()
    for sender_input, _, _, _ in data:
        mask = sender_input[3]
        if mask.sum().item() != 1:
            continue
        count_roles_D1[mask.argmax()] += 1
    entropy_roles_D1 = entropy_list(count_roles_D1.values())
    print("Entropy of roles on D_1:", entropy_roles_D1)

    # when we want to filter on entity 8, we mean the 8+1th most frequent
    # entity in the train set. This matches how the qualitative analysis
    # script analysis.py numbers unique entities.
    # here, we use different code because we don't use interactions (obtained
    # by putting data through model) but iterate thru data loader.
    count_arguments = data.count_arguments(trainDL)
    idx2prop = {idx: prop for idx, (prop, c) in enumerate(count_arguments.most_common())}
    prop2idx = {p: i for i, p in idx2prop.items()}
    ####### GENERALISATION
    # Are there tuples not seen at test time?
    arg_tuples_train = data.count_arg_tuples(prop2idx, trainDL)
    arg_tuples_test = data.count_arg_tuples(prop2idx, testDL)
    in_test_not_in_train = {T: c for T, c in arg_tuples_test.items() if T not in arg_tuples_train}
    in_test_in_train = {T: c for T, c in arg_tuples_test.items() if T in arg_tuples_train}
    # 4 is all 0
    #  print("8, 5, -1")
    #  data.print_filtered_examples(idx2prop, entities=(8,5,4), use_nltk=True)
    #  print("-1, 8, 5")
    #  data.print_filtered_examples(idx2prop, entities=(4,8,5), use_nltk=True)
    #  print("65, 8, 4")
    #  for roleset in [431, 410, 804, 805, 1532, 327]:
    #      data.print_filtered_examples(idx2prop, prop2idx, roleset=roleset,
    #          use_nltk=True, skip_shown=False)
    data.print_filtered_examples(idx2prop, prop2idx, entities=7, use_nltk=True)
    ###### Similar annotations?
    # Created & existed before seems to be mutually exclusive.
    # That's true in most cases: 
    # Counter({(1, 3): 1284, (1, 2): 198, (3, 1): 158, (0, 1): 98, (0, 0): 95, (0, 2): 82, (2, 2): 79, (0, 3): 76, (1, 1): 69, (2, 1): 58, (3, 0): 21, (2, 0): 11, (1, 0): 7})
    #  count_created_existed_before = Counter()
    #  for idx, prop in idx2prop.items():
    #      count_created_existed_before[(prop[4], prop[7])] += 1
    #  import pdb; pdb.set_trace()
