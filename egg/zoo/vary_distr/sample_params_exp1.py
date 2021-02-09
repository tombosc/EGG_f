import numpy as np

def choice(l):
    return np.random.choice(l)

params = {
    'batch_size': choice([64, 128, 256]),
    'random_seed': choice([1, 2, 3, 4, 5]),
    'max_len': choice([10]),
    'lr': choice([0.0001, 0.0003, 0.001]),
    'vocab_size': choice([10, 20, 40]),
    'sender_entropy_coef': choice([0.05, 0.1, 0.2, 0.3]),
    'length_coef': choice([0.05, 0.1, 0.2, 0.3]),
    'min_distractors': choice([2, 3]),
    'max_distractors': choice([6, 10]),
    'n_features': choice([4, 6]),
    'max_value': choice([4, 6]),
    'embed_dim': choice([16]),
    'n_examples': 1024*5,
}

if __name__ == "__main__":
    cmd = ''
    for k, v in params.items():
        cmd += '--' + k + '=' + str(v) + ' '
    print(cmd)
