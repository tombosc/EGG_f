#!/bin/bash

QUALITATIVE=0
CONCATENABILITY=0  # and transitivity, too
ROLE_PRED=0
CONTEXT_INDEPENDENCE=0
GENERALISATION_OOD=0
GENERALISATION_IID=0

for N in res_proto_min*_adam/*
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
	if [ $CONCATENABILITY = 1 ]; then
		echo "Concatenability $DIR"
		echo "python -m egg.zoo.vd_reco.eval_probas_concat $DIR/best.tar test"
		python -m egg.zoo.vd_reco.eval_probas_concat $DIR/best.tar test 
	fi
	FN_ANALYSIS="${DIR}/analysis" 
	if [ $QUALITATIVE = 1 ] && [ ! -f "$FN_ANALYSIS" ]; then
	# if [ $QUALITATIVE = 1 ]; then
		echo "Quali $DIR"
		python -m egg.zoo.vd_reco.analyze $DIR/best.tar > $FN_ANALYSIS
	fi
	FN_ROLE_PRED="${DIR}/role_predict.json"
	# if [ $ROLE_PRED = 1 ]; then
	if [ $ROLE_PRED = 1 ] && [ ! -f "$FN_ROLE_PRED" ]; then
		python -m egg.zoo.vd_reco.predict_role_1obj $DIR/best.tar

	fi
	FN_CONTEXT_INDEPENDENCE="${DIR}/context_ind_train.json"
	# if [ $CONTEXT_INDEPENDENCE = 1 ]; then
	if [ $CONTEXT_INDEPENDENCE = 1 ] && [ ! -f "$FN_CONTEXT_INDEPENDENCE" ]; then
		echo "Context indep"
		# python -m egg.zoo.vd_reco.context_independence --split train $DIR/best.tar
		python -m egg.zoo.vd_reco.context_independence --split train $DIR/best.tar 
		cat $FN_CONTEXT_INDEPENDENCE
	fi

	FN_GENERALISATION_OOD="${DIR}/test_generalization.json"
	# if [ $GENERALISATION_OOD = 1 ]; then
	if [ $GENERALISATION_OOD = 1 ] && [ ! -f "$FN_GENERALISATION_OOD" ]; then
		echo "Generalisation OoD"
		python -m egg.zoo.vd_reco.test_generalization $DIR/best.tar 
		cat $FN_GENERALISATION_OOD
	fi
	FN_GENERALISATION_IID="${DIR}/test_generalization_iid.json"
	if [ $GENERALISATION_IID = 1 ] && [ ! -f "$FN_GENERALISATION_IID" ]; then
		echo "Generalisation IID"
		python -m egg.zoo.vd_reco.test_generalization $DIR/best.tar --in-distribution
		cat $FN_GENERALISATION_IID
	fi
	# tail -n 2 $N | head -n 1
	# echo "done $FN_ANALYSIS"
done
