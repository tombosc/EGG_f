# python -m egg.zoo.vary_distr.play --print_validation_events \
#     --data 'dd' --data.seed=1 --min_distractors 1 --max_distractors 15 --max_value 5 --n_features 5 \
#     --batch_size=256  --lr=0.001 --n_epochs=2000 --optimizer 'adam'\
#     --vocab_size 20 --max_len 10 \
#     --sender_marg_entropy_coef 0.0 --length_coef 0.0 --log_length=False \
#     --hp.seed=1 --sender_type='tfm' --embed_dim 8 --n_layers 2 --n_heads 4 --lstm_hidden=30 --receiver_type='simple' --share_embed False --embedder='cat' \
#     -C "" \
#     --no_cuda

# example of command for re-training the receiver. Note that
# * --random_seed can be different than the one used for initializing the
# original model
# * --n_epochs should be larger than the first n_epochs, because training
# "continues" starting from the last epoch.
python -m egg.zoo.vary_distr.play --retrain_receiver_shuffled --load_from_checkpoint /media/Docs/data/eggs_exps/data_dd_sd_1_nex_5120_mdis_1_Mdis_15_Mv_5_nft_5_P_3_sd_1_Ed_8_lenC_0.0_lenCep_0_SdrHC_0.0_SdrmargHC_0.0_lstmhid_30_Sdrtype_tfm_Rcvtype_simple_Eder_mean_nH_8_nlay_2_lrsched_F_bs_256_O_adam_lr_0.001_Vsz_20_Mlen_10/100.tar --n_epochs 2100 --no_cuda --seed=1
