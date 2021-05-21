#!/bin/bash

module load anaconda/3 pytorch/1.7
# conda activate gra
# echo $CONDA_PREFIX

source $HOME/venv_gra/bin/activate

NAME="proto_2_adam"
SEED=$1
N_RUNS=$2
EXP_BASENAME="${NAME}_sd${SEED}"
TMP_EXP_DIR="${SLURM_TMPDIR}/bosctom/${EXP_BASENAME}"
# the backup/final exp dir is not seed specific
FINAL_EXP_DIR="/network/tmp1/bosctom/EGG_f/res_${NAME}"
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

# I think NCCL flag is not necessary anymore, since I've added --no_distribute
NCCL_IB_DISABLE=1 python gs_vd_reco.py $TMP_EXP_DIR egg/zoo/vd_reco/hyperparam_grid/${NAME}.json $SEED $N_RUNS --cuda --variable_length --backup $FINAL_EXP_DIR
do_at_the_end
