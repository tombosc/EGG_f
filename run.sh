#!/bin/sh

python -m egg.zoo.vary_distr.play --mode 'rf' --print_validation_events --batch_size=64 --random_seed=42 --no_cuda --max_len 9 --lr=0.0003 --n_epochs=10 > log
