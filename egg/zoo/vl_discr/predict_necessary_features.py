""" Predict the features that are necessary to identify the target. It is not
trivial to find an architecture...
"""
import torch
from .data_readers import init_simple_data, SimpleData
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter, defaultdict


class NPropPredictor(nn.Module):
    def __init__(self, dim_emb, dim_ff, dropout,
            n_properties, n_values, n_max_distractors, 
            initial_target_bias, 
            n_layers=3, n_head=8):
        super(NPropPredictor, self).__init__()
        self.idx_offset = nn.Parameter(torch.arange(0, n_properties) *
                n_values, requires_grad=False)
        self.value_embedding = nn.Embedding(1 + n_values*n_properties, dim_emb)
        self.mark_features = nn.Parameter(torch.randn(size=(1, 1, n_properties, dim_emb)))
        self.mark_objects = nn.Parameter(torch.randn(size=(1, n_max_distractors+1, 1, dim_emb)))
        activation = 'gelu'
        encoder_layer = nn.TransformerEncoderLayer(dim_emb, n_head, dim_ff, dropout, activation)
        self.n_properties = n_properties
        self.encoder_prop = nn.TransformerEncoder(encoder_layer, n_layers)
        self.encoder_obj = nn.TransformerEncoder(encoder_layer, n_layers)
        self.n_max_distractors = n_max_distractors
        self.predict_n_necessary = nn.Sequential(
            nn.Linear(n_properties*dim_emb, n_properties),
        )

    def forward(self, x):
        mask = (x == 0)
        x = self.value_embedding(x + self.idx_offset.unsqueeze(0))
        x += self.mark_features
        x += self.mark_objects
        N = self.n_max_distractors + 1
        nP = self.n_properties
        bs = x.size(0)
        d = x.size(-1)
        # first, treat each property independently
        x = x.transpose(2, 1).reshape((bs * nP, N, d)) 
        mask_rs = mask.transpose(2, 1).reshape((bs * nP, N))
        y = self.encoder_prop(
            x.transpose(0, 1),
            src_key_padding_mask=mask_rs,
        ).view((N, bs, nP, d))
        y = self.encoder_obj(
            y[0].transpose(0, 1), #  nP, bs, d
            #  src_key_padding_mask=mask_rs,
        ).view((nP, bs, d))
        y_cat = y.transpose(0, 1).reshape((bs, -1))  # bs, nP×d
        pred_n = self.predict_n_necessary(y_cat)
        # n_prop, bs, d
        return pred_n

c = SimpleData.Settings(n_examples=4000, max_value=4, n_features=4)
all_data, train_loader, valid_loader, test_loader = init_simple_data(
    c, 0, 32, 1024, True, num_workers=0,
)

m = NPropPredictor(32, 200, 0.2,
    c.n_features, c.max_value, c.max_distractors, 0,
    n_layers=1, n_head=16,
)

#  optimizer = torch.optim.Adam(m.parameters(), lr=3e-4)#, betas=(0.9, 0.99))
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.9))

def nec_feat_to_bool(nec_features, n_features):
    """ Transform a matrix (batch dim: rows) with necessary features (or -1)
    into another matrix of 0 and 1
    Example: turn row (3 1) into (0 1 0 1)
    """
    bs = nec_features.size(0)
    targets = torch.zeros((bs, n_features)).bool()
    for i, nec_f in enumerate(nec_features):
        for f in nec_f:
            if f >= 0:
                targets[i, f] = 1
    return targets.to(nec_features.device)

loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

def eval_pred(logits, tgts):
    probas = F.sigmoid(logits)
    thresh = probas > 0.5
    is_eq = (tgts == thresh)
    bin_per_K = Counter()
    n_bin_per_K = Counter()
    K = tgts.sum(1)
    for E, k in zip(is_eq, K):
        kint = int(k.item())
        bin_per_K[kint] += E.float().sum().item()
        n_bin_per_K[kint] += E.size(0)
    print("k=1", bin_per_K[1] / n_bin_per_K[1])
    print("k=2", bin_per_K[2] / n_bin_per_K[2])
    return is_eq.float().mean().item()

for i in range(200):
    train_loss, val_loss, test_loss = [], [], []
    for sender_in, _, _ in train_loader:
        inputs, nec_feats = sender_in
        tgts = nec_feat_to_bool(nec_feats, c.n_features)
        pred_n = m(inputs)
        optimizer.zero_grad()
        loss = loss_func(pred_n, tgts.float())
        loss_mean = loss.mean()
        train_loss.append(loss_mean.item())
        loss_mean.backward()
        optimizer.step()
    print(f"loss i={i}", np.mean(train_loss))
        
    with torch.no_grad():
        for sender_in, _, _ in test_loader:
            inputs, nec_feats = sender_in
            tgts = nec_feat_to_bool(nec_feats, c.n_features)
            pred_n = m(inputs)
            #  if i==50:
            #      import pdb; pdb.set_trace()
            loss = loss_func(pred_n, tgts.float())
            eval_pred(pred_n, tgts.float())
            #  print("H", eval_pred(pred_n, tgts.float()))
            test_loss.append(loss.mean().item())
        print(f"test loss i={i}", np.mean(test_loss))
