python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=256 --random_seed=1 --max_len 5 --lr=0.001 --n_epochs=1500 --vocab_size 40 --sender_entropy_coef 0.0 --length_coef 0.0 --min_distractors 1 --max_distractors 5 --max_value 5 --n_features 3 --sender_type='tfm' --embed_dim 12 --n_layers 1 --n_heads 2 --log_length=False --lstm_hidden=30 --receiver_type='simple' --share_embed False -C "" --optimizer 'adam'

# example of command for re-training the receiver. Note that
# * --random_seed can be different than the one used for initializing the
# original model
# * --n_epochs should be larger than the first n_epochs, because training
# "continues" starting from the last epoch.
# python -m egg.zoo.vary_distr.play --load_from_checkpoint /media/Docs/data/eggs_exps/sd_3_nex_5120_mdis_1_Mdis_5_Mvalue_5_nft_3_sd_3_Ed_8_lenC_0.03_lenCep_100_SdrHC_0.0_SdrmargHC_1e-05_lstmhid_30_Sdrtype_tfm_Rcvtype_simple_Eder_cat_nH_8_nlay_2_lrsched_F_C_MH_bs_128_O_adam_lr_0.001_Vsz_20_Mlen_10/2000.tar --retrain_receiver_deduped --n_epochs 2100 --no_cuda --random_seed=2
