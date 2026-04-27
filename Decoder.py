import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
import torch

class Decoder_block(nn.Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int):
        super(Decoder_block, self).__init__()
        self.self_mha = MultiHeadAttention(num_heads, d_model, d_k)
        self.cross_mha = MultiHeadAttention(num_heads, d_model, d_k)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor = None):
        mha_output = self.self_mha(x, x, x, mask=mask)

        x = self.norm1(x+mha_output)

        cross_attn_output = self.cross_mha(x, encoder_output, encoder_output, mask=mask)
        x = self.norm2(x+cross_attn_output)

        ffn_output = self.ffn(x)
        x = self.norm3(x+ffn_output)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers: int, numheads: int, d_model: int, d_k: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [Decoder_block(numheads, d_model, d_k) for _ in range(num_layers)]
        )
        
    def forward(self, x:torch.Tensor, encoder_output: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, encoder_output, mask=mask)
        return x