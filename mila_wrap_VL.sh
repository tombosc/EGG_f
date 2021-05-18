#!/bin/bash

module load anaconda/3 pytorch/1.7
# conda activate gra
# echo $CONDA_PREFIX

source $HOME/venv_gra/bin/activate

NAME="proto_1_adam"
SEED="0"
N_RUNS="10"
EXP_BASENAME="${NAME}_sd${SEED}"
TMP_EXP_DIR="${SLURM_TMPDIR}/bosctom/${EXP_BASENAME}"
# the backup/final exp dir is not seed specific
FINAL_EXP_DIR="res_${NAME}"
mkdir -p $TMP_EXP_DIR
mkdir -p $EXP_DIR
echo $TMP_EXP_DIR $EXP_DIR
python gs_vd_reco.py $TMP_EXP_DIR egg/zoo/vd_reco/hyperparam_grid/${NAME}.json $SEED $N_RUNS --cuda --variable_length --backup $FINAL_EXP_DIR
