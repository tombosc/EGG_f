#!/bin/bash

# conda activate postpro

# For example, 
# (postpro) # ./examine_ptb.sh 1931 9

FN=$1
DIR="${FN:0:2}"
SENT_ID=$2
echo $FN $DIR $SENT_ID
python -c "from nltk.corpus import ptb; print(ptb.sents(fileids='WSJ/${DIR}/WSJ_${FN}.MRG')[${SENT_ID}])"
