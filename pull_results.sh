#!/bin/sh

rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_min1_adam/*" res_proto_min1_adam --exclude="*/202*"
rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_min2_adam/*" res_proto_min2_adam --exclude="*/202*"
rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_min2B_adam/*" res_proto_min2B_adam --exclude="*/202*"

# Saved this command line as an example for include with recursivity
# rsync -am --include "val_avg_loss_per_ts.npy" --include "results_sample*.json" --include "log.txt" --include='*/' --include='*/*/' --exclude='*' --exclude='*/*' mila:"/network/tmp1/bosctom/vae_pretraining_encoder/$DIR" .
