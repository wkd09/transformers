import torch.nn as nn
import torch
from MultiHeadAttention import MultiHeadAttention

class Encoder_block(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int):
        super(Encoder_block, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_model, d_k)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 1. MHA + Residual + Norm
        mha_output = self.mha(x, x, x, mask=mask)
        x = self.norm1(x + mha_output)
        
        # 2. FFN + Residual + Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers:int, num_heads:int, d_model:int, d_k:int):
          super(Encoder, self).__init__()
          self.layers = nn.ModuleList(
               [Encoder_block(num_heads, d_model, d_k) for _ in range(num_layers)]
          )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
          x = layer(x, mask=mask)
        return x