#!/bin/sh

# rsync -am mila:"/home/mila/b/bosctom/EGG_f/{fixed,var}*" res_reproduction_ent_min/
# rsync -am mila:"/home/mila/b/bosctom/EGG_f/res_var_Hmin_smorms3" .
# rsync -am mila:"/home/mila/b/bosctom/EGG_f/res_var_Hmin_b_smorms3" .
# OLD directory
# rsync -am mila:"/home/mila/b/bosctom/EGG_f/res_proto_1_adam" .
# NEW
# rsync -avm --no-r mila:"/network/tmp1/bosctom/EGG_f/res_proto_3_adam/*" res_proto_3_adam
# Get specific interactions
# rsync -avm mila:"/network/tmp1/bosctom/EGG_f/res_proto_3_adam/67ca2f*I" res_proto_3_adam --exclude='*/*.tar'
# rsync -avm mila:"/network/tmp1/bosctom/EGG_f/res_proto_3_adam/5751d3*I" res_proto_3_adam --exclude='*/*.tar'
# rsync -avm mila:"/home/mila/b/bosctom/res_proto_1sh_adam/cbf095ae2fc7fe04d*" res_proto_1sh_adam
# rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1r_adam/*" res_proto_1r_adam --exclude="*/2021*"
# rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1rNS_adam/*" res_proto_1rNS_adam --exclude="*/2021*"
# rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1rNP_adam/*" res_proto_1rNP_adam --exclude="*/2021*"
# rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1flat_adam/*" res_proto_1flat_adam --exclude="*/2021*"
# rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1flatNP_adam/*" res_proto_1flatNP_adam --exclude="*/2021*"
# rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1rNada_adam/*" res_proto_1rNada_adam --exclude="*/2021*"
# rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1rV_adam/*" res_proto_1rV_adam --exclude="*/2021*"
rsync -avm mila:"/home/mila/b/bosctom/EGG_f/res_proto_1rVvaryH_adam/*" res_proto_1rVvaryH_adam --exclude="*/2021*"

# Saved this command line as an example for include with recursivity
# rsync -am --include "val_avg_loss_per_ts.npy" --include "results_sample*.json" --include "log.txt" --include='*/' --include='*/*/' --exclude='*' --exclude='*/*' mila:"/network/tmp1/bosctom/vae_pretraining_encoder/$DIR" .
