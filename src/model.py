import torch
import torch.nn as nn
import numpy as np
import utils

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg):
        
        src_len = src.shape[1]
        trg_len = trg.shape[1]
        batch_size = src.shape[0]
        
        src_pos = torch.stack([torch.tensor(range(1,src_len+1))] * batch_size) 
        trg_pos = torch.stack([torch.tensor(range(1,trg_len+1))] * batch_size)
        # trg = trg[::-1]
        # trg_pos = trg_pos[::-1]
        
        enc_output = self.encoder(src, src_pos)
        dec_output = self.decoder(trg, trg_pos, src, enc_output)
        
        return dec_output
    
    def predict(self):
        pass
    
