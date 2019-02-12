import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_key, dropout=0.1):
        super().__init__()
        self.temp = np.sqrt(d_key)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        score = torch.bmm(q, k.transpose(1, 2)) / self.temp

        if mask is not None:
            score = score.masked_fill(mask, -np.inf)

        att_weights = self.dropout(self.softmax(score))
        output = torch.bmm(att_weights, v)

        return output, att_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_x, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_x = d_x
        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_k

        self.w_x2q = nn.Linear(d_x, n_head * self.d_q)
        self.w_x2k = nn.Linear(d_x, n_head * self.d_k)
        self.w_x2v = nn.Linear(d_x, n_head * self.d_v)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.layer_norm = nn.LayerNorm(d_x)

        self.fc = nn.Linear(n_head * d_v, d_x)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # B: batchsize, L: sequnece length
        B, L_Q, _ = Q.size()
        B, L_K, _ = K.size()
        B, L_V, _ = V.size()

        residual = Q

        q = self.w_x2q(Q).view(B, L_Q, self.n_head, self.d_k)
        k = self.w_x2k(K).view(B, L_K, self.n_head, self.d_k)
        v = self.w_x2v(V).view(B, L_V, self.n_head, self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_q)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)

        mask = mask.repeat(self.n_head, 1, 1)
        output, att_weights = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, B, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(B, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, att_weights