import json
import glob
import shutil
import os
import argparse
import numpy as np
import subprocess
from egg.zoo.vd_reco.train import main
from contextlib import redirect_stdout
from hashlib import sha256

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str)
    parser.add_argument('config_json', type=str)
    parser.add_argument('seed', type=int)
    parser.add_argument('n_runs', type=int)
    args = parser.parse_args()

    print("Running with seed", args.seed)
    np.random.seed(args.seed)

    with open(args.config_json, 'r') as f:
        hp = json.load(f)
    # first, check that no config or the config is already there, but not a
    # different config.
    configs = glob.glob(os.path.join(args.exp_dir, "*.json"))
    if configs:
        with open(configs[0], 'r') as f:
            json_existing_config = json.load(f)
        with open(args.config_json, 'r') as f:
            json_arg_config = json.load(f)
        if json_arg_config != json_existing_config:
            print("Existing config", json_existing_config)
            print("Argument config", json_arg_config)
            raise ValueError("Directory {} already contains a json config, and"
                    "it is different than the passed config.".format(args.exp_dir))

    shutil.copy(args.config_json, args.exp_dir)

    for i in range(10):
        #  cmd = ['bash', '-c', '"conda activate egg; python -m egg.zoo.vd_reco.train']
        #  cmd = ["python", "-m", "egg.zoo.vd_reco.train"]
        cmd = ['--no_cuda']
        for k, v in hp.items():
            chosen_v = np.random.choice(v).item()
            print(k, chosen_v, type(chosen_v))
            if type(chosen_v) == bool:
                if chosen_v:
                    cmd.append("--" + k)
            else:
                cmd.append("--" + k)
                cmd.append(str(chosen_v))
        H = sha256(''.join(cmd).encode('utf8')).hexdigest()[:32]
        
        fn_output = os.path.join(args.exp_dir, H)
        if os.path.exists(fn_output):
            print("Already ran: " + " ".join(cmd))
            continue
        print(' '.join(cmd))
        print(H)

        with open(fn_output, 'w') as f_out:
            with redirect_stdout(f_out):
                main(cmd)
