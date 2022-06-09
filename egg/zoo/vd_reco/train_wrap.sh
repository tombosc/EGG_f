#!/bin/bash

# Run random search
conda activate egg_env
echo $CONDA_PREFIX

NAME="proto_min1_adam"
SEED=$1
N_RUNS=$2
EXP_BASENAME="${NAME}_sd${SEED}"
# if running on a cluster, it's good to write data on the local node's drive.
# once training is done, copy the results (models, logs) at another location, 
# cf do_at_the_end() and exit_script().
# if you're not running on a cluster, it doesn't matter much.
TMP_EXP_DIR="/tmp/egg_exps_tmp"  # or ${SLURM_TMPDIR}/...
FINAL_EXP_DIR="~/EGG/res_${NAME}"
mkdir -p $TMP_EXP_DIR
mkdir -p $FINAL_EXP_DIR
echo $TMP_EXP_DIR 

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
