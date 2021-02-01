# python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=256 --random_seed=3 --max_len 5 --lr=0.01 --n_epochs=50 --vocab_size 10 --sender_entropy_coef 0.1 --length_coef 0.1 --min_distractors 3 --max_distractors 10 --n_features 4

python -m egg.zoo.vary_distr.play --print_validation_events --batch_size=256 --random_seed=3 --max_len 5 --lr=0.001 --n_epochs=450 --vocab_size 10 --sender_entropy_coef 0.1 --length_coef 0.3 --min_distractors 3 --max_distractors 10 --n_features 4
