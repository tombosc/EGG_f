import json
import glob
import shutil
import os
import argparse
import numpy as np
import subprocess
from egg.zoo.vd_reco.train import main
from egg.zoo.vd_reco.train_vl import main as main_vl
from contextlib import redirect_stdout
from hashlib import sha256

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str)
    parser.add_argument('config_json', type=str)
    parser.add_argument('seed', type=int)
    parser.add_argument('n_runs', type=int)
    parser.add_argument('--backup', type=str, help='Backup to directory after each run.')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--variable_length', default=False, action='store_true')
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

    n_run = 0
    while n_run < args.n_runs:
        #  cmd = ['bash', '-c', '"conda activate egg; python -m egg.zoo.vd_reco.train']
        #  cmd = ["python", "-m", "egg.zoo.vd_reco.train"]
        cmd = ['--no_distributed']
        if not args.cuda:
            cmd.append('--no_cuda')

        for k, v in hp.items():
            chosen_v = np.random.choice(v).item()
            if type(chosen_v) == bool:
                if chosen_v:
                    cmd.append("--" + k)
            else:
                cmd.append("--" + k)
                cmd.append(str(chosen_v))
        H = sha256(''.join(cmd).encode('utf8')).hexdigest()[:32]
        checkpoint_dir = os.path.join(args.exp_dir, H + '_I')
        cmd += ['--checkpoint_dir', checkpoint_dir]
        #print(H)
        
        #continue
        fn_output = os.path.join(args.exp_dir, H)
        if args.backup:
            backup_fn_output = os.path.join(args.backup, H)
        if (os.path.exists(fn_output) or 
                (args.backup and os.path.exists(backup_fn_output))):
            print("Already ran:", H)
            continue
        print(' '.join(cmd))
        print(H)
        # need to write something in the backupfile! otherwise, several scripts
        # can start to compute the same run!
        if args.backup:
            with open(backup_fn_output, 'w') as f:
                f.write('Computation in process...')
        with open(fn_output, 'w') as f_out:
            with redirect_stdout(f_out):
                if args.variable_length:
                    main_vl(cmd)
                else:
                    main(cmd)
        if args.backup:
            shutil.copy(fn_output, backup_fn_output)
        n_run += 1
