from positional_encode import positional_encoding
from Decoder import Decoder
from Encoder import Encoder

import torch

class Transformer:
  def __init__(self, num_layers: int, num_heads: int, d_model: int, d_k: int, dropout: float):
    self.encoder = Encoder(num_layers, num_heads, d_model, d_k)
    self.decoder = Decoder(num_layers, num_heads, d_model, d_k)
    self.positional_encoding = positional_encoding(d_model, dropout)

  def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None):
    # PE
    src = self.positional_encoding(src)
    tgt = self.positional_encoding(tgt)
  
    # Encoder
    encoder_output = self.encoder(src, mask=src_mask)

    # Decoder
    output  = self.decoder(tgt, encoder_output, mask=tgt_mask)
    return output