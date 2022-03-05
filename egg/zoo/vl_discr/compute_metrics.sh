#!/bin/bash

VARIOUS=1

for N in resD_hp_decontext_2_tangled/*
do
	DIR="${N}_I"
	if [ ${N: -2} = "_I" ]; then
		continue
	fi
	if [ ! -d $DIR ]; then
		continue
	fi
	# theoretically we shoud look at best, not last, but this is roughly similar here
	LAST_TEST_DICT=`tail -n 2 $N | head -n 1`
	# extract length
	echo $DIR
	LENGTH=`python -c 'import sys; import json; d = json.loads(sys.argv[1]); print(d["length"])' "$LAST_TEST_DICT"`
	# the following lines deactivate computation of metrics when 
	# length is too large or too small.
	# HOWEVER, it is simplistic and operate on the length of the last test epoch, 
	# not the best one. 
	# 3 runs out of 100 were excluded wrongfully and I had to detect that in the
	# statistical_tests.py script (there were NaNs).
	# a more brutal way to avoid that is simply to remove these checks.
	if (( $(echo "$LENGTH < 2" |bc -l) )); then
		echo "avg length $LENGTH too small"
		continue
	fi
	if (( $(echo "$LENGTH > 7" |bc -l) )); then
		echo "avg length $LENGTH too large"
		continue
	fi
	echo "Will analyze (average length=${LENGTH})"
	FN_ANALYSIS="${DIR}/analysis.json" 
	if [ $VARIOUS = 1 ] && [ ! -f "$FN_ANALYSIS" ]; then
		echo "Analyze $DIR"
		python -m egg.zoo.vl_discr.compute_concat $DIR/best.tar
	else
		cat $FN_ANALYSIS |json_pp
	fi
done
