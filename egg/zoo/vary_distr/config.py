from dataclasses import dataclass, field
from .data_readers import Data, data_selector
from .architectures import Hyperparameters, EGGParameters, GlobalParams

from simple_parsing.helpers import Serializable
from typing import List
import os
import json
from types import SimpleNamespace



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
       'data': data_cls.Config.load(path_to("data.json")),
       'hp': Hyperparameters.load(path_to("hp.json")),
       'core': EGGParameters.load(path_to("core.json")),
       'glob': global_params,
    }

