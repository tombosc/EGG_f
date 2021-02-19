# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F
import egg.core as core
from egg.core.baselines import EMABaseline, MeanBaseline
from .utils import set_torch_seed
from .config import EGGParameters, Hyperparameters


class Embedder(nn.Module):
    """ For each object, embed its features differently for each feature
    position.
    """
    def __init__(self, n_features, dim_embed, max_value, pooling):
        super().__init__()
        self.padding_value = 0
        # since 0 is a padding value, all the actual features go from 1 to
        # max_value+1 (not included). For each feature, the embedding will be
        # different; that's why we need self.embed_vector.
        self.embed_dim = dim_embed
        self.embeddings = nn.Embedding(1 + n_features * max_value, dim_embed)
        self.embed_vector = (torch.arange(n_features) * max_value).view(1, 1, n_features)
        assert(pooling in ['mean', 'cat'])
        if pooling == 'mean':
            self.output_dim = dim_embed
        elif pooling == 'cat':
            self.output_dim = dim_embed * n_features
        self.pooling = pooling

    def forward(self, x, ret_first_row):
        """ if ret_first_row=True, return the encoding of the first object
        only. Else, return the entire matrix.
        """
        bs, max_L, n_features = x.size()
        mask = torch.all((x == self.padding_value), dim=2)
        device = x.device
        embedded = self.embeddings(x + self.embed_vector.to(device))
        if self.pooling == 'mean':
            x = embedded.mean(2)
        elif self.pooling == 'cat':
            x = embedded.view(bs, max_L, -1)
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        if ret_first_row:
            return x[:, 0], mask
        else:
            return x, mask
        return d

class SimpleSender(nn.Module):
    """ Encode the first row of the matrix. 
    """
    def __init__(self, encoder, output_dim):
        super().__init__()
        self.encoder = encoder
        self.output = nn.Linear(encoder.embed_dim, output_dim)

    def forward(self, x):
        x, _ = self.encoder(x, ret_first_row=True)
        out = self.output(x)
        return out

class TransformerSender(nn.Module):
    """ Pragmatic: the target is "contextualized" before being encoded.
    """
    def __init__(self, encoder, embed_dim, output_dim, ff_dim, max_objects,
            n_heads, n_layers, encoder_side):
        super().__init__()
        self.encoder = encoder
        self.encoder_side = encoder_side
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=ff_dim, 
        )
        layer_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=layer_norm,
        )
        self.output = nn.Linear(embed_dim, output_dim)
        self.output_dim = output_dim
        # target_embed will mark the target (the first row of sender_input)
        self.target_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, ret_first_row=False):
        if ret_first_row:
            raise ValueError("Unused parameter: hack")
        embedded, mask = self.encoder(x, ret_first_row=False)
        if self.encoder_side:
            embedded[:, 0] = embedded[:, 0] + self.target_embed
        out = self.transformer_encoder(
            src=embedded.transpose(0, 1),
            src_key_padding_mask=mask,
        ).transpose(0, 1) # (bs, lmax, n_feat)
        out = self.output(out)
        if self.encoder_side:
            return out[:, 0]
        else:
            return out, mask

class DiscriReceiverEmbed(nn.Module):
    """ A basic discriminative receiver, like DiscriReceiver, but which expect
    integer (long) input to embed, not one-hot encoded.
    """
    def __init__(self, n_features, input_dim, n_hidden, n_embeddings, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(input_dim, n_hidden)

    def forward(self, x, _input):
        # mask: (bs, L) set to False iff all features are padded
        _input, mask = self.encoder(_input, ret_first_row=False)
        _input = self.fc1(_input).tanh()
        dots = torch.matmul(_input, torch.unsqueeze(x, dim=-1)).squeeze()
        dots = dots.masked_fill(mask, -float('inf'))
        return dots

def create_encoder(
    data,
    hp,
):
    sender_embedder = Embedder(
        n_features=data.n_features,
        dim_embed=hp.embed_dim,
        max_value=data.max_value,
        pooling=hp.embedder,
    )

    if hp.sender_type == 'simple':
        sender = SimpleSender(
            sender_embedder,
            output_dim=hp.lstm_hidden,
        )
    elif hp.sender_type == 'tfm':
        if hp.embedder == 'cat':
            in_dim = data.n_features * hp.embed_dim
        elif hp.embedder == 'mean':
            in_dim = hp.embed_dim

        sender = TransformerSender(
            sender_embedder,
            embed_dim=in_dim,
            output_dim=hp.lstm_hidden,
            ff_dim=4*hp.embed_dim,
            max_objects = data.max_distractors + 1,
            n_heads=hp.n_heads,
            n_layers=hp.n_layers,
            encoder_side=True,
        )
    else:
        raise ValueError()
    return sender, sender_embedder


def create_game(
    core_params: EGGParameters,
    data,
    hp: Hyperparameters,
    loss,
    shuffle_message=False,
    dedup_message=False,
):
    """ Create all the modules.
    """
    set_torch_seed(hp.seed)
    sender, sender_embedder = create_encoder(data, hp)

    sender = core.RnnSenderReinforce( 
        sender,
        vocab_size=core_params.vocab_size,
        embed_dim=hp.embed_dim,
        hidden_size=hp.lstm_hidden,
        cell='lstm',
        max_len=core_params.max_len,
        condition_concat=True,
        always_sample=True,
    )

    if hp.share_embed:
        receiver_embedder = sender_embedder
    else:
        receiver_embedder = Embedder(
            n_features=data.n_features,
            dim_embed=hp.embed_dim,
            max_value=data.max_value,
            pooling=hp.embedder,
        )

    if hp.receiver_type == 'simple':
        receiver = DiscriReceiverEmbed(
                n_features=data.n_features,
                input_dim=receiver_embedder.output_dim,
                n_hidden=hp.lstm_hidden,
                n_embeddings=data.max_value,
                encoder=receiver_embedder,
        )
        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=core_params.vocab_size,
            embed_dim=hp.embed_dim,
            hidden_size=hp.lstm_hidden,
            cell='lstm',
        )
    elif hp.receiver_type == 'att':
        receiver = AttentionReceiver(
                input_encoder=receiver_embedder,
                embed_dim=receiver_embedder.output_dim,
                hidden_size=hp.lstm_hidden,
                max_msg_len=core_params.max_len,
                n_max_objects=data.max_distractors,
                vocab_size=core_params.vocab_size,
        )
    elif hp.receiver_type == 'tfm':
        if hp.embedder == 'cat':
            in_dim = data.n_features * hp.embed_dim
        elif hp.embedder == 'mean':
            in_dim = hp.embed_dim

        receiver_embedder = TransformerSender(
            receiver_embedder,
            embed_dim=in_dim,
            output_dim=hp.lstm_hidden,
            ff_dim=4*hp.embed_dim,
            max_objects = data.max_distractors + 1,
            n_heads=hp.n_heads,
            n_layers=hp.n_layers,
            encoder_side=False,
        )
        receiver = DiscriReceiverEmbed(
            n_features=data.n_features,
            input_dim=receiver_embedder.output_dim,
            n_hidden=hp.lstm_hidden,
            n_embeddings=data.max_value,
            encoder=receiver_embedder,
        )

        receiver = core.RnnReceiverDeterministic(
            receiver,
            vocab_size=core_params.vocab_size,
            embed_dim=hp.embed_dim,
            hidden_size=hp.lstm_hidden,
            cell='lstm',
        )

    else:
        raise ValueError()

    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=hp.sender_entropy_coef,
        sender_marginal_entropy_coeff=hp.sender_marg_entropy_coef,
        receiver_entropy_coeff=0.,
        length_cost=hp.length_coef,
        log_length=hp.log_length,
        baseline_type=MeanBaseline,
        shuffle_message=shuffle_message,
        dedup_message=dedup_message,
    )
    return game

class FCAttention(nn.Module):
    def __init__(self, key_dim, query_value_dim, dot_product):
        super().__init__()
        self.dot_product = dot_product
        self.attn_combine = nn.Linear(key_dim + query_value_dim, query_value_dim)
        if dot_product:
            # project query
            self.query_projection = nn.Linear(query_value_dim, key_dim)
        else:
            self.attn_weights = nn.Linear(key_dim + query_value_dim, 1)

    def forward(self, q, M, mask):
        """ Shapes:
            * q=(bs, d')
            * M=(bs, L, d)
            * mask=(bs, d')
        Softmax over dimension 1 of M.
        """
        #  print("DBG Att q={}, M={}".format(q.size(), M.size()))
        L_max = M.size(1)
        if self.dot_product:
            project_q = self.query_projection(q).unsqueeze(2)  # (bs, d, 1)
            # (bs, L, d) x (bs, d, 1) 
            attn_weights_unnorm = torch.bmm(M, project_q)  # (bs, L, 1)
        else:
            expanded_q = q.unsqueeze(1).expand(-1, L_max, -1) # (bs, L, d')
            cat = torch.cat((M, expanded_q), 2) # (bs, L, d + d')
            attn_weights_unnorm = self.attn_weights(cat) # (bs, L, 1)
        #  print("S", mask.size(), attn_weights_unnorm.size())
        attn_weights_unnorm = attn_weights_unnorm.masked_fill(mask.unsqueeze(2),
                -float('inf'))
        attn_weights = F.softmax(attn_weights_unnorm, dim=1) # (bs, L, 1)
        # (bs, 1, L) x (bs, L, d)
        attn_applied = torch.bmm(attn_weights.transpose(2, 1), M).squeeze(1) # (bs, d)
        #  print("DBG", attn_applied.size(), q.unsqueeze(1).size())
        output = torch.cat((attn_applied, q), 1)
        output = self.attn_combine(output)
        #  output = F.relu(output)
        #  print("DBG", output.size())
        return output, attn_weights.squeeze(2)

class AttentionReceiver(nn.Module):
    def __init__(self, input_encoder, embed_dim, hidden_size, max_msg_len, n_max_objects, vocab_size):
        super().__init__()
        self.input_encoder = input_encoder
        self.embedding_msg = nn.Embedding(vocab_size+1, embed_dim)
        self.msg_encoder = nn.LSTMCell(embed_dim, hidden_size)
        self.hidden_size = hidden_size
        self.attention = FCAttention(embed_dim, hidden_size, dot_product=True)
        self.max_msg_len = max_msg_len
        self.fc_out = nn.Linear(hidden_size, n_max_objects)

    def msg_mask(self, messages):
        zero_mask = messages == 0
        lengths = (zero_mask.cumsum(dim=1) > 0).sum(dim=1)

    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size, device=device)

    def forward(self, msg, input=None, lengths=None):
        #  print("H1", msg.size(), input.size())
        bs = msg.size(0)
        msg_mask = self.msg_mask(msg)
        #  print(input.size())
        device = input.device
        # embed objects
        #  print("Pre in", input.size())
        input, input_mask = self.input_encoder(input, ret_first_row=False)
        msg_embed = self.embedding_msg(msg)
        h = self.init_hidden(device).expand(bs, -1)
        c = self.init_hidden(device).expand(bs, -1)
        hidden_states = []
        attn_weights = []
        for i in range(self.max_msg_len + 1):
            att_out, att_w = self.attention(h, input, input_mask)
            #  attn_w = attn_w.squeeze(2)
            hidden_states.append(att_out)
            attn_weights.append(att_w)
            #  print("unrolling: h={}, c={}, msg_embed={}".format(
            #      attn_h.size(), c.size(), msg_embed.size(),
            #  ))
            if i == self.max_msg_len:
                break
            h, c = self.msg_encoder(msg_embed[:, i], (att_out, c))

        hidden_states = torch.stack(hidden_states).transpose(1, 0)
        attn_weights = torch.stack(attn_weights).transpose(1, 0)
        # (bs, L_max, d)
        lengths = torch.clamp(lengths, max=self.max_msg_len)
        # variant 1: use the last hidden state
        #  output = hidden_states[torch.arange(bs), lengths]#.relu()
        #  output = self.fc_out(output)
        #  output = output.masked_fill(input_mask, -float('inf'))

        # variant 2: don't care: simply unroll to max_len and pick last
        #  output = hidden_states[:, -1]
        #  output = self.fc_out(output)
        #  output = output.masked_fill(input_mask, -float('inf'))

        # variant 3: output is directly the attn_h
        output = attn_weights[torch.arange(bs), lengths]  # TODO
        #  print("OUT", output.size())
        logits = torch.zeros(output.size(0), device=device)
        entropy = logits
        return output, logits, entropy
