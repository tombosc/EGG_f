import numpy as np

def choice(l):
    return np.random.choice(l)

embed_dim = choice([4, 8])
if embed_dim == 8:
    n_heads = choice([4, 8])
else:
    n_heads = 4

params = {
    'batch_size': choice([64, 128, 256]),
    'random_seed': choice([1, 2, 3, 4, 5]),
    'max_len': choice([10]),
    'lr': choice([0.0003, 0.001]),
    'vocab_size': choice([20]),
    # 'sender_entropy_coef': choice([0.05, 0.1, 0.2, 0.3]),
    # 'length_coef': choiCE([0.05, 0.1, 0.2, 0.3]),
    'length_coef': choice([1e-2, 3e-2, 1e-1, 3e-1]),
    'length_coef_epoch': choice([100]),
    'sender_entropy_coef': choice([0]),  # deprecated!
    'sender_marg_entropy_coef': choice([1e-5,1e-4,1e-3,1e-2]),
    'lstm_hidden': choice([30, 40]),
    'min_distractors': choice([1]),
    'max_distractors': choice([5]),
    'n_features': choice([3]),
    'max_value': choice([5]),
    'embed_dim': embed_dim,
    'sender_type': 'tfm',
    'receiver_type': 'simple',
    'n_heads': n_heads,
    'n_layers': choice([2]),
    'n_examples': 1024*5,
    'embedder': 'cat',
    'C': 'MH',
}

if __name__ == "__main__":
    cmd = ''
    for k, v in params.items():
        cmd += '--' + k + '=' + str(v) + ' '
    print(cmd)
