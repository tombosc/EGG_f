import json
import os
import argparse
import numpy as np
import subprocess
from egg.zoo.vd_reco.train import main
from contextlib import redirect_stdout
from hashlib import sha256

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('n_runs', type=int)
    args = parser.parse_args()

    print("Running with seed", args.seed)
    np.random.seed(args.seed)

    hp_fn = 'egg/zoo/vd_reco/hyperparam_grid/gs_main_variable.json'
    with open(hp_fn, 'r') as f:
        hp = json.load(f)
    hp['bits_r'] = [4]

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
        fn_output = os.path.join('res_vary_d_reco_var_T/' + H)
        if os.path.exists(fn_output):
            print("Already ran: " + " ".join(cmd))
            continue
        print(' '.join(cmd))
        print(H)

        with open(fn_output, 'w') as f_out:
            with redirect_stdout(f_out):
                main(cmd)
