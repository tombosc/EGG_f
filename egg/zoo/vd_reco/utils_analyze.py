import re
import numpy as np
import json


def extract_config_from_namespace(line):
    """ Very crude: given a line containing str(dict), recover dict.
    """
    # TODO replace that or cross-check with other json files in the exp dir?
    line = line.rstrip(')\n')
    config = re.findall("([a-zA-Z_0-9]+)='?([^,'()]*)'?", line)      
    return {k: v for k, v in config}             


def listdict2dictlist(L):
    """ Given a list of dict with identical keys, turn that into a dict of numpy arrays.
    """
    keys = L[0].keys()
    res = {}
    for k in keys:
        res[k] = np.asarray([l[k] for l in L])
    return res


def parse_log(fn):
    """ Parse a log in a file fn.
    """
    with open(fn, 'r') as f:
        cfg = extract_config_from_namespace(f.readline())
        train_logs, test_logs = [], []
        for line in f.readlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if parsed['mode'] == 'train':
                train_logs.append(parsed)
            elif parsed['mode'] == 'test':
                test_logs.append(parsed)
        train_logs = listdict2dictlist(train_logs)
        test_logs = listdict2dictlist(test_logs)
        return cfg, train_logs, test_logs

def common_filter_runs(df):
    """ Both for producing tables & plots, we need to use the same filtering.
    Voluntarily does not take any filter parameters.
    """
    min_filter, max_filter = 2.0, 7.0
    print(f"Filter out runs with avg length < {min_filter}, > {max_filter}")
    len_before_filtering = len(df)
    df = df[df['final_length'] > 2.0]
    df = df[df['final_length'] < 7.0]
    print(f"Before = {len_before_filtering}, after = {len(df)}")
    print("Final lengths:")
    print(df[['final_length_1.0', 'final_length_2.0', 'final_length_3.0']].mean())
    return df
