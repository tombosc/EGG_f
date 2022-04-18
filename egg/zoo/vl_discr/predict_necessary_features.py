""" Predict the features that are necessary to identify the target. It is not
trivial to find an architecture...
"""
import torch
from .data_readers import init_simple_data, SimpleData
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter, defaultdict
from egg.core.smorms3 import SMORMS3


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
            #  nn.Linear(n_properties*dim_emb, 3*dim_emb),
            #  nn.ReLU(),
            #  nn.Linear(3*dim_emb, n_properties),
            #  nn.Linear(n_properties*dim_emb, n_properties),
            nn.Linear(dim_emb, 1),
        )
        self.predict_similar = nn.Sequential(
            nn.Linear(dim_emb, n_properties),
        )
        self.ln = nn.LayerNorm(dim_emb)
        self.ln2 = nn.LayerNorm(dim_emb)
        self.ln3 = nn.LayerNorm(dim_emb)

    def forward(self, x):
        """ x: (bs, N, nP, d)
        """
        mask = (x == 0)
        x = self.value_embedding(x + self.idx_offset.unsqueeze(0))
        #  x += self.mark_features
        x += self.mark_objects
        N = self.n_max_distractors + 1
        nP = self.n_properties
        bs = x.size(0)
        d = x.size(-1)
        # first, treat each property independently
        x = x.transpose(2, 1).reshape((bs * nP, N, d))
        mask_rs = mask.transpose(2, 1).reshape((bs * nP, N))
        y = self.encoder_prop(
            x.transpose(0, 1),  # N, bs*nP, d
            src_key_padding_mask=mask_rs,
        ).view((N, bs, nP, d))
        #  y = self.ln(y[0])
        y = y[0] + self.mark_features.squeeze(1)
        similar = self.predict_similar(y.detach())
        # tmp dbug
        #  y = self.encoder_prop(self.ln2(y)).view((N, bs, nP, d))
        y = self.encoder_obj(
            y.transpose(0, 1), #  nP, bs, d
        )
        #  y = self.encoder_obj(self.ln3(y))
        # joint prediction
        #  y_cat = y.transpose(0, 1).reshape((bs, -1))  # bs, nP×d
        #  pred_n = self.predict_n_necessary(y_cat)
        # inidependent prediction for each prop
        y_cat = y.transpose(0, 1).reshape((bs * nP, d))
        pred_n = self.predict_n_necessary(y_cat).view((bs, nP))
        # n_prop, bs, d
        return pred_n, similar


c = SimpleData.Settings(n_examples=4000, max_value=8, ood=True,
        disentangled=True, max_distractors=4, n_features=5)
all_data, train_loader, valid_loader, test_loader = init_simple_data(
    c, 0, 128, 1024, True, num_workers=0,
)

m = NPropPredictor(128, 200, 0.2,
    c.n_features, c.max_value, c.max_distractors, 0,
    n_layers=2, n_head=8,
)

#  optimizer = torch.optim.Adam(m.parameters(), lr=3e-4)#, betas=(0.9, 0.99))
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.99))
#  optimizer = SMORMS3(m.parameters(), lr=1e-3)

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
loss_aux = torch.nn.CrossEntropyLoss(reduction='none')

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
    for k in bin_per_K.keys():
        print(f"k={k}", bin_per_K[k] / n_bin_per_K[k])
    return is_eq.float().mean().item()

for i in range(300):
    train_loss, val_loss, test_loss = [], [], []
    train_loss_aux = []
    for sender_in, _, _ in train_loader:
        inputs, nec_feats = sender_in
        tgts = nec_feat_to_bool(nec_feats, c.n_features)
        pred_n, pred_similar = m(inputs)
        n_similar = (inputs[:,0].unsqueeze(1) == inputs)[:, 1:].sum(1)
        optimizer.zero_grad()
        loss_a = loss_aux(pred_similar.transpose(1, 2), n_similar)
        loss_a_mean = loss_a.mean()
        train_loss_aux.append(loss_a_mean.item())
        loss = loss_func(pred_n, tgts.float())
        loss_mean = loss.mean()
        train_loss.append(loss_mean.item())
        (loss_mean + loss_a_mean).backward()
        optimizer.step()
    print(f"loss i={i}", np.mean(train_loss), " aux ", np.mean(train_loss_aux))
        
    with torch.no_grad():
        for sender_in, _, _ in test_loader:
            inputs, nec_feats = sender_in
            tgts = nec_feat_to_bool(nec_feats, c.n_features)
            pred_n, _ = m(inputs)
            #  if i==50 or i==200:
            if i==200:
                import pdb; pdb.set_trace()
            loss = loss_func(pred_n, tgts.float())
            eval_pred(pred_n, tgts.float())
            #  print("H", eval_pred(pred_n, tgts.float()))
            test_loss.append(loss.mean().item())
        print(f"test loss i={i}", np.mean(test_loss))
