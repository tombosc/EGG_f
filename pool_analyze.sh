#!/bin/sh

for DIR in $EGG_EXPS_ROOT/*tfm_nheads_4*; do
    echo $DIR
    python -m egg.zoo.vary_distr.pool_analyze $DIR
done
