import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg):
        super().__init__()
        self.letter_embedding = nn.Embedding(cfg['voc_len'], cfg['dim'])
        # self.pos_embed = nn.Embedding(cfg['pos_dim'], cfg['dim'])  # position embedding
        self.norm = LayerNorm(cfg['dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])

        E = cfg['dim']
        maxlen = 25
        position_enc = torch.tensor([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)], dtype=torch.float32)

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])  # dim 2i+1
        self.pos_embed = nn.Embedding.from_pretrained(position_enc)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)  # (S,) -> (B, S)

        e = self.letter_embedding(x) + self.pos_embed(pos)
        return self.drop(self.norm(e))

    # def positional_encoding(self, inputs, maxlen,masking=False):
    #      N, T, E = inputs.shape
    #      # position indices
    #      position_ind = torch.range(0, T, device=inputs.device).expand(N, -1)
    #          #tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)
    #
    #      # First part of the PE function: sin and cos argument
    #      position_enc = torch.tensor([
    #          [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
    #          for pos in range(maxlen)], device=inputs.device, dtype=torch.float32)
    #
    #      # Second part, apply the cosine to even columns and sin to odds.
    #      position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
    #      position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])  # dim 2i+1
    #      # position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)
    #
    #      # lookup
    #      outputs = tf.nn.embedding_lookup(position_enc, position_ind)
    #
    #      # masks
    #      if masking:
    #          outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
    #
    #      return tf.to_float(outputs)


class MultiHeadAttention(nn.Module):

    def __init__(self, cfg):

        super(MultiHeadAttention, self).__init__()
        self.n_heads = cfg['n_heads']
        self.dim = cfg['outdim']
        # self.dropout_rate = att_drop_rate

        self.proj_q = nn.Linear(cfg['qdim'], cfg['outdim'])
        self.proj_k = nn.Linear(cfg['kdim'], cfg['outdim'])
        self.proj_v = nn.Linear(cfg['vdim'], cfg['outdim'])
        self.drop = nn.Dropout(cfg['dropout_rate'])
        self.scores = None
        self.ln = LayerNorm(cfg['dim'])

    def mask(self, inputs, key_masks=None, type=None):
        """Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (h*N, T_q, T_k)
        key_masks: 3d tensor. (N, 1, T_k)
        type: string. "key" | "future"
        e.g.,
        >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
        >> key_masks = tf.constant([[0., 0., 1.],
                                    [0., 1., 1.]])
        >> mask(inputs, key_masks=key_masks, type="key")
        array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
           [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
           [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
           [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
        """
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            pass
        #     key_masks = tf.to_float(key_masks)
        #     key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
        #     key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        #     outputs = inputs + key_masks * padding_num
        # elif type in ("q", "query", "queries"):
        #     # Generate masks
        #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
        #
        #     # Apply masks to inputs
        #     outputs = inputs*masks
        elif type in ("f", "future", "right"):
            diag_vals = torch.ones_like(inputs[0,0, :, :])  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0) # (T_q, T_k)
            future_masks = tril.expand(inputs.shape[0],inputs.shape[1], -1, -1)  # (N, T_q, T_k)

            paddings = torch.ones_like(future_masks) * padding_num
            outputs = torch.where(future_masks.eq(0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")

        return outputs

    def forward(self, query, key, value,  causality=False, mask=None):
        q, k, v = self.proj_q(query), self.proj_k(key), self.proj_v(value)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        if causality:
            scores = self.mask(scores, type='future')
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        outputs = query + h
        outputs = self.ln(outputs)
        return outputs


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['dim'], cfg['dim_ff'])
        self.fc2 = nn.Linear(cfg['dim_ff'], cfg['dim'])
        self.ln = LayerNorm(cfg['dim'])
        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.ln(self.fc2(gelu(self.fc1(x))) + x)

# class Decoder
