from dataclasses import dataclass, field
from simple_parsing.helpers import Serializable
from typing import List
import os


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
        'validation': 'val', 'batch_size': 'bs', 'length': 'len',
        'coef': 'C', 'entropy': 'H', 'sender': 'sendr', 'receiver': 'recv',
        'hidden': 'hid', 'examples': 'ex', 'features': 'ft', 
        'seed': 'sd', 'optimizer': 'O', 'pretrained': 'pre', 'train': 'tr',
        'test': 'ts', 'output': 'out', 'linear': 'lin', 'embeddings': 'E',
        'embed': 'E', 'dim': 'd', 'improvement': 'imp', 'precision': 'prc',
        'frozen': 'frz', 'target': 'tgt', 'format': 'fmt', 'l2_loss_coef':
        'l2', 'standardize': 'std', 'epochs': 'ep', 'loss': 'L', 'name': '',
        'distractors': 'dis', 'min': 'm', 'max': 'M', 
        'vocab': 'V', 'size': 'sz',
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

