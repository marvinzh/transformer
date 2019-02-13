import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_K, dropout_rate=0.1):
        super().__init__()
        self.temp = np.sqrt(d_K)
        self.dropout_rate = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, mask=None):
        score = torch.bmm(Q, K.transpose(1, 2)) / self.temp

        if mask is not None:
            score = score.masked_fill(mask, -np.inf)

        att_weights = self.dropout_rate(self.softmax(score))
        output = torch.bmm(att_weights, V)

        return output, att_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_X, d_K, d_V, dropout_rate=0.1):
        super().__init__()

        H = n_head
        self.d_X = d_X

        self.d_Q = d_K
        self.d_K = d_K
        self.d_V = d_V
        
        self.w_x2q = nn.Linear(d_X, n_head * self.d_Q)
        self.w_x2k = nn.Linear(d_X, n_head * self.d_K)
        self.w_x2v = nn.Linear(d_X, n_head * self.d_V)

        self.attention = ScaledDotProductAttention(self.d_K)
        self.layer_norm = nn.LayerNorm(d_X)
        self.fc = nn.Linear(n_head * d_V, d_X)
        self.dropout_rate = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask=None):
        H = self.n_head
        # B: batchsize, L: sequnece length
        B, L_Q, _ = Q.size()
        B, L_K, _ = K.size()
        B, L_V, _ = V.size()

        residual = Q

        Qs = self.w_x2q(Q).view(B, L_Q, H, self.d_Q)
        Ks = self.w_x2k(K).view(B, L_K, H, self.d_K)
        Vs = self.w_x2v(V).view(B, L_V, H, self.d_V)

        # (B*H, L, D)
        # B: batchsize, H: # of heads, L: sequnece length, D: embedding dimension
        Qs = Qs.permute(2, 0, 1, 3).contiguous().view(-1, L_Q, self.d_Q)
        Ks = Ks.permute(2, 0, 1, 3).contiguous().view(-1, L_K, self.d_K)
        Vs = Vs.permute(2, 0, 1, 3).contiguous().view(-1, L_V, self.d_V)


        mask = mask.repeat(H, 1, 1)
        output, att_weights = self.attention(Qs, Ks, Vs, mask=mask)


        output = output.view(H, B, L_Q, self.d_V)
        # (B, L_Q, H * D)
        # concat heads at each timestep together
        output = output.permute(1, 2, 0, 3).contiguous().view(B, L_Q, -1)

        output = self.dropout_rate(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, att_weights