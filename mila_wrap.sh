#!/bin/bash

module load anaconda/3 pytorch/1.5
# conda activate gra
# echo $CONDA_PREFIX
# about virtualenv's variables, see https://stackoverflow.com/a/38645983
source $HOME/venv_gra/bin/activate

N_EPOCH=2000

for i in `seq 1 10`; do
    TMP_DIR=$(mktemp -d -p $SLURM_TMPDIR)
    echo "$TMP_DIR" 

    PARAMS=$(python -m egg.zoo.vary_distr.sample_params_exp1_tfm)
    # PARAMS=$(python -m egg.zoo.vary_distr.replicate_exp1_tfm)
    # execute script using the temporary exp directory
    EGG_EXPS_ROOT="$TMP_DIR" python -m egg.zoo.vary_distr.play $PARAMS --no_cuda --n_epochs="$N_EPOCH"

    # analyse reuslts
    REL_EXP_DIR=$(ls $TMP_DIR)

    for n in 50 100 200 500 1000 1500 $N_EPOCH; do
        EGG_EXPS_ROOT="$TMP_DIR" python -m egg.zoo.vary_distr.analyze $REL_EXP_DIR/validation/interactions_epoch"$n" > "$TMP_DIR/$REL_EXP_DIR/analyze_res_ep$n"
    done

    # copy to the proper exps directory
    cp -dR $TMP_DIR/* $EGG_EXPS_ROOT
    rm -r $TMP_DIR
done
