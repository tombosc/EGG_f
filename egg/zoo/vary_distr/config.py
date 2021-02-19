import argparse
from simple_parsing import ArgumentParser, Serializable
from dataclasses import dataclass, fields, asdict
from .data_readers import data_selector
import egg.core as core
from simple_parsing.helpers import Serializable
from typing import List
import os
import json
from types import SimpleNamespace

 
@dataclass
class GlobalParams(Serializable):
    """ Parameters that should be parsed before other parameters are parsed,
    because they define how they should be parsed/what to parse. 
    Examples:
   -  models can have different hyperparameters, so the model type should be
    specified before.
    - same for datasets.
    """
    data: str = 'id'  

    def __post_init__(self):
        assert(self.data in ['dd', 'id'])


@dataclass
class RetrainParams(Serializable):
    """ Parameters that are only parsed when load_from_checkpoint is used.
    """
    retrain_receiver: bool = False
    shuffled: bool = False
    deduped: bool = False

    def __post_init__(self):
        a = self.retrain_receiver
        b = self.shuffled
        c = self.deduped
        assert(not(a and b) and not(a and c) and not(b and c))

    def get_dict_dirname(self):
        d = self.__dict__.copy()
        
        def delete_or_replace(a, b):
            if d[a] == False:
                del d[a]
            else:
                d[b] = d[a]
                del d[a]

        delete_or_replace('shuffled', 'RSh')
        delete_or_replace('deduped', 'RDe')
        delete_or_replace('retrain_receiver', 'R')
        return d


@dataclass
class EGGParameters(Serializable):
    random_seed: int
    batch_size: int
    checkpoint_dir: str
    optimizer: str
    lr: float
    vocab_size: int
    max_len: int

    @classmethod
    def from_argparse(cls, args):
        """ Assumes that args is a namespace containing all the field names.
        """
        d = [getattr(args, f.name) for f in fields(cls)]
        return cls(*d)

    def fill_namespace(self, args):
        for k, v in asdict(self).items():
            setattr(args, k, v)

    def get_dict_dirname(self):
        d = self.__dict__.copy()
        del d['random_seed']
        del d['checkpoint_dir']
        return d


@dataclass
class Hyperparameters(Serializable):
    seed: int = 0
    embed_dim: int = 30
    validation_batch_size: int = 0
    length_coef: float = 0.
    length_coef_epoch: int = 0  # if n > 0, increment by 0.01 every n epochs
    log_length: bool = False
    sender_entropy_coef: float = 0.
    sender_marg_entropy_coef: float = 0.
    lstm_hidden: int = 30  
    sender_type: str = 'simple'  # 'simple' or 'tfm'
    receiver_type: str = 'simple' # 'simple' or 'att'
    embedder: str = 'mean'
    # tfm specific
    n_heads: int = 4
    n_layers: int = 2
    lr_sched: bool = False
    grad_norm: float = 0
    C: str = ''  # a simple comment
    share_embed: bool = False
    
    def __post_init__(self):
        assert(self.embed_dim > 0)
        assert(self.lstm_hidden > 0)

    def get_dict_dirname(self):
        d = self.__dict__.copy()
        if d['sender_type'] == 'simple':
            for u in ['n_heads', 'n_layers']:
                del d[u]
        del d['validation_batch_size']
        if d['grad_norm'] == 0:
            del d['grad_norm']
        if not d['log_length']:
            del d['log_length']
        if not d['share_embed']:
            del d['share_embed']
        if d['C'] == '':
            del d['C']




def represent_list_as_str(l):
    s = ''
    for e in l:
        s += e + '-'
    return s[:-1]


def represent_dict_as_str(d):
    """ Compute a string representation of a dataclass instance.

    In particular, abbreviate some terms and call `dataclass.get_dict_dirname`
    to shorten the representation.
    """
    abbreviations = {
        'validation': 'val', 'value': 'v', 'batch_size': 'bs', 'length': 'len',
        'coef': 'C', 'entropy': 'H', 'sender': 'Sdr', 'receiver': 'Rcv',
        'hidden': 'hid', 'examples': 'ex', 'features': 'ft', 
        'seed': 'sd', 'optimizer': 'O', 'pretrained': 'pre', 'train': 'tr',
        'test': 'ts', 'output': 'out', 'linear': 'lin', 'embeddings': 'E',
        'embed': 'E', 'dim': 'd', 'improvement': 'imp', 'precision': 'prc',
        'frozen': 'frz', 'target': 'tgt', 'format': 'fmt', 'l2_loss_coef':
        'l2', 'standardize': 'std', 'epochs': 'ep', 'epoch': 'ep', 'loss': 'L', 'name': '',
        'heads': 'H', 'head': 'H', 'layers': 'lay', 
        'distractors': 'dis', 'min': 'm', 'max': 'M', 
        'vocab': 'V', 'size': 'sz',
        'retrain_receiver_shuffled': 'RSh',
        'retrain_receiver_deduped': 'RDe',
        'patience': 'P',
    }
    val = {'True': 'T', 'False': 'F'}
    s = ''
    try:
        items = d.get_dict_dirname().items()
    except AttributeError:
        items = d.__dict__.items()
    for k, v in items:
        for key, short_key in abbreviations.items():
            k = k.replace(key, short_key)
        if type(v) == list:
            v = represent_list_as_str(v)
        else:
            v = str(v)
            if v in val:
                v = val[v]
        if v is not None and v != []:
            k = k.replace('_', '')
            s += k + '_' + str(v) + '_'
    return s[:-1]


def compute_exp_dir(exps_root, configs):
    s = ''
    for v in configs.values():
        s += represent_dict_as_str(v) + '_'
    s = s[:-1]
    return os.path.join(exps_root, s)


def stem_path(path):
    base_fn = os.path.basename(path)
    base, ext = os.path.splitext(base_fn)
    return base

def save_configs(configs, exp_dir):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    for config_name, config in configs.items():
        full_config_fn = os.path.join(exp_dir, config_name + '.json')
        config.save(path=full_config_fn)

def load_configs(exp_dir):
    if not os.path.exists(exp_dir):
        raise ValueError("Missing directory {}".format(exp_dir))

    #  def read_json(filename):
    #      with open(os.path.join(exp_dir, filename), 'r') as f:
    #          return json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    #  return {
    #     'data': read_json('data.json'),
    #     'hp': read_json('hp.json'),
    #     'core': read_json('core.json'),
    #  }
    def path_to(filename):
        return os.path.join(exp_dir, filename)
    global_params = GlobalParams.load(path_to("glob.json"))
    data_cls = data_selector[global_params.data]
    return {
       'glob': global_params,
       'data': data_cls.Config.load(path_to("data.json")),
       'hp': Hyperparameters.load(path_to("hp.json")),
       'core': EGGParameters.load(path_to("core.json")),
    }


def get_config(params):
    # this is quite messy, because parsing is in 2 phases:
    # 1. parse GlobalParameters indicating what Dataset class is used, whether
    #    we load a checkpoint, (and possibly in the future: what model is used,
    #    etc.).
    # 2. parse Dataset parameters parsed in phase 1, etc. 
    # it's actually more complex, because when we continue training, we ignore
    # most of the command line arguments, since we read config from the
    # experiment directory.
    # phase 1:
    parser_1 = ArgumentParser()
    parser_1.add_arguments(GlobalParams, dest='glob')
    parser_1.add_argument('--load_from_checkpoint', default='')
    args_1, left_over = parser_1.parse_known_args()
    parser_2 = ArgumentParser()
    checkpoint_fn = args_1.load_from_checkpoint
    if checkpoint_fn:
        # read config files
        if not os.path.exists(checkpoint_fn):
            raise ValueError("Missing checkpoint: {}".format(checkpoint_fn))
        exp_dir = os.path.dirname(checkpoint_fn)
        configs = load_configs(exp_dir)
        data_cls = data_selector[configs['glob'].data]
        print(args_1)
        print("data_cls", data_cls)
        print(configs['data'])
        print(configs['glob'])
        parser_2.add_arguments(RetrainParams, dest='retrain')
    else:
        data_cls = data_selector[args_1.glob.data]
        parser_2.add_arguments(data_cls.Config, dest='data')
    # When we load a checkpoint, we sometimes retrain part of the architecture
    # (see --retrain...). In that case, we want to control the seed. That's why
    # we use Hyperparameters.
    parser_2.add_arguments(Hyperparameters, dest='hp')
    parser_2.add_argument('--print_validation_events', action='store_true')
    args_2 = core.init(parser_2, left_over)
    # merge the namespaces
    args_2.glob = args_1.glob
    if checkpoint_fn:
        configs['hp'].seed = args_2.hp.seed
        configs['retrain'] = args_2.retrain
        # set global opts that are read by core.trainer
        configs['core'].fill_namespace(args_2)
        args_2.load_from_checkpoint = checkpoint_fn  # very important! has
        #  already been parsed during phase 1, but core.init has its own.
    else:
        core_params = EGGParameters.from_argparse(args_2)
        configs = {  # order matters, must match that of load_configs
            'glob': args_2.glob,
            'data': args_2.data,
            'hp': args_2.hp,
            'core': core_params,
        }
    return configs, data_cls, checkpoint_fn
