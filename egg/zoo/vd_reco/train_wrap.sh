#!/bin/bash

# Run random search. Take 2 arguments:
SEED=$1 
N_RUNS=$2

# in the paper:
# NAME is "proto_min1_adam", "proto_min2_adam" or "proto_min2B_adam"
# corresponds to a file in egg/zoo/vd_reco/hyperparam_grid/$NAME.json
NAME="proto_min1_adam"  
# if running on a cluster, it's good to write data on the local node directly.
# that's TMP_EXP_DIR
TMP_EXP_DIR="/tmp/egg_exps_tmp"  # or ${SLURM_TMPDIR}/...
# once training is done, copy the results (models, logs) at another location 
# FINAL_EXP_DIR, cf do_at_the_end() and exit_script().
FINAL_EXP_DIR="$HOME/EGG/res_${NAME}"
# if you're not running on a cluster, it doesn't matter much.
EXP_BASENAME="${NAME}_sd${SEED}"
mkdir -p $TMP_EXP_DIR
mkdir -p $FINAL_EXP_DIR
echo "Temporary directory: $TMP_EXP_DIR"
echo "Final directory: $FINAL_EXP_DIR"
echo "Conda environment: $CONDA_PREFIX"

do_at_the_end() {
    cp -dr $TMP_EXP_DIR/* $FINAL_EXP_DIR/
}

exit_script() {
    echo "Preemption signal, saving myself"
    trap - SIGTERM # clear the trap
    # Optional: sends SIGTERM to child/sub processes
    do_at_the_end
}

trap exit_script SIGTERM

python -m egg.zoo.vd_reco.random_search $TMP_EXP_DIR egg/zoo/vd_reco/hyperparam_grid/${NAME}.json $SEED $N_RUNS --cuda --backup $FINAL_EXP_DIR
do_at_the_end
