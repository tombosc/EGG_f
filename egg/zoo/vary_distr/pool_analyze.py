import json
import os
import argparse

def last_epoch_performance(fn):
    with open(fn, 'r') as f:
        for line in f:
            pass
        line = json.loads(line)
        assert(line['mode'] == 'test')
        return line['acc']

def print_lengths(fn):
    lengths = []
    first = True
    with open(fn, 'r') as f:
        lines = [json.loads(line) for line in f]
        for i, line in enumerate(lines):
            if type(line) == dict:
                print("{:.2f}±{:.2f} ({}); δ={:.2f}".format(
                    line['mean_data'], line['std_data'], line['n_datapoints'],
                    line['mean_data'] - lines[0]['mean_data'],
                ))
            first = False
    print()

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("file_name", type=str)
    args = parser.parse_args()

    exps_root = os.environ["EGG_EXPS_ROOT"]
    logs = os.path.join(exps_root, args.exp_dir, 'logs.txt')
    acc = last_epoch_performance(logs)
    print(args.exp_dir)
    if acc > 0.3:
        print("Acc={}".format(acc))
        results = os.path.join(exps_root, args.exp_dir, args.file_name)
        print_lengths(results)
