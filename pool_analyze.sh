#!/bin/sh

for DIR in $EGG_EXPS_ROOT/sd*; do
    python -m egg.zoo.vary_distr.pool_analyze $DIR analyze_res_ep2000
done
