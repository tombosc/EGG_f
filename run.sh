# python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=256 --random_seed=3 --max_len 5 --lr=0.01 --n_epochs=50 --vocab_size 10 --sender_entropy_coef 0.1 --length_coef 0.1 --min_distractors 3 --max_distractors 10 --n_features 4 --sender_type=simple --embed_dim 16

# Works! 2 layers, 4 heads:
# - stagnates until epoch 70,
# - 0.41 around 100 epochs,
# - 0.736 around 200 epochs.
# Seems to be a bit stuck. Maybe lr is too high still? Note that it is 0.001 whereas 0.01 worked for the simpler architecture!
# Now if embed_dim=8: 200 ep: 0.70
# python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=256 --random_seed=3 --max_len 5 --lr=0.001 --n_epochs=450 --vocab_size 10 --sender_entropy_coef 0.1 --length_coef 0.1 --min_distractors 3 --max_distractors 10 --n_features 4 --sender_type=tfm --embed_dim 16
# With 0.15 length_coef, embed_dim16 100 epochs: 0.16, 200: 0.65, 300: 0.67 (stuck)
# same but with embed_dim=64: 100 ep: 0.18
# same but with embed_dim=8: 100 ep: 0.17, 200: 0.63, 300:0.7 (stuck around 0.69)
# embed_dim=8, nlay=3 instead of 2: 100 ep: 0.17, 200: 0.63, 300:0.7 (stuck around 0.69)
# bs=128, lr=0.001: 150 epochs, nothing, same for bs=128, lr=0.0003
# now, without sender_entropy_coef, bs=256 and lr=0.001: NO, 150 epochs nothing
# now, without sender_entropy_coef=0.5, bs=256 and lr=0.001: NO, 150 epochs nothing
# 0.5, 0.5, lr3e-4, bs256: no
# 1.5 0.5 lr=1e-3 bs=256: no
# python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=37 --random_seed=1 --max_len 5 --lr=0.001 --n_epochs=1500 --vocab_size 40 --sender_entropy_coef 0.0 --length_coef 0.01 --min_distractors 3 --max_distractors 10 --max_value 5 --n_features 5 --sender_type='tfm' --embed_dim 16 --n_layers 1 --n_heads 2 --log_length=False --lstm_hidden=50 --receiver_type='simple' --share_embed False -C "newerD" --optimizer 'sgd'
# Works: ep100: 33, 30; ep 200: 82, 83 acc
# python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=256 --random_seed=1 --max_len 5 --lr=0.001 --n_epochs=1500 --vocab_size 40 --sender_entropy_coef 0.0 --length_coef 0.0 --min_distractors 1 --max_distractors 3 --max_value 5 --n_features 3 --sender_type='tfm' --embed_dim 12 --n_layers 1 --n_heads 2 --log_length=False --lstm_hidden=30 --receiver_type='simple' --share_embed False -C "newerD" --optimizer 'adam'
# works: ep100: 25/26, 200: 58,61; 300: 89, 91
# python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=256 --random_seed=1 --max_len 5 --lr=0.001 --n_epochs=1500 --vocab_size 40 --sender_entropy_coef 0.0 --length_coef 0.0 --min_distractors 1 --max_distractors 5 --max_value 5 --n_features 3 --sender_type='tfm' --embed_dim 12 --n_layers 1 --n_heads 2 --log_length=False --lstm_hidden=30 --receiver_type='simple' --share_embed False -C "newerD" --optimizer 'adam'
# n_features 4 doesn't wor yet
