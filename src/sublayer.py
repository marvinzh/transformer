import torch
import torch.nn as nn
import numpy as np
from attention import MultiHeadAttention

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, dropout_rate=0.1):
        super().__init__()
        self.conv_1 = nn.Conv1d(d_in, d_hidden, 1)
        self.conv_2 = nn.Conv1d(d_hidden, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout_rate = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.conv_2(torch.relu(self.conv_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout_rate(output)
        output = self.layer_norm(output + residual)
        return output
    
class EncoderLayer(nn.Module):
    def __init__(self, d_input, d_inner, n_head, d_K, d_V, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        # multi-head attention
        self.mh_att = MultiHeadAttention(n_head, d_input, d_K, d_V, dropout_rate=dropout_rate)
        # position wise feed foward nets
        self.pos_nn = PositionWiseFeedForward(d_input, d_inner, dropout_rate=dropout_rate)

    def forward(self, X, non_pad_mask=None, mh_att_mask=None):
        att_out, att_weights = self.mh_att(X, X, X, mask=mh_att_mask)
        att_out = att_out * non_pad_mask

        output = self.pos_nn(att_out) * non_pad_mask

        # (B, L, D), (B*H, L, L)
        return output, att_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_input, d_inner, n_head, d_K, d_V, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.input_mh_att = MultiHeadAttention(n_head, d_input, d_K, d_V, dropout_rate=dropout_rate)
        self.enc_mh_att = MultiHeadAttention(n_head, d_input, d_K, d_V, dropout_rate=dropout_rate)
        self.pos_nn = PositionWiseFeedForward(d_input, d_inner, dropout_rate=dropout_rate)

    def forward(self, X, enc_outputs, non_pad_mask=None, input_att_mask=None, enc_att_mask=None):
        self_att_out, self_att_weights = self.input_mh_att(X, X, X, mask=input_att_mask)
        self_att_out = self_att_out * non_pad_mask
        
        enc_att_out, enc_att_weights = self.enc_mh_att(self_att_out, enc_outputs, enc_outputs, mask=enc_att_mask)
        enc_att_out = enc_att_out * non_pad_mask
        
        output = self.pos_nn(enc_att_out) * non_pad_mask
        
        return output, self_att_weights, enc_att_weights