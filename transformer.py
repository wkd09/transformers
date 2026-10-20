from positional_encode import positional_encoding
from Decoder import Decoder
from Encoder import Encoder

import torch
import torch.nn as nn

class Transformer(nn.Module):
  def __init__(self, num_layers: int, num_heads: int, d_model: int, d_k: int, dropout: float):
    super().__init__()
    self.encoder = Encoder(num_layers, num_heads, d_model, d_k)
    self.decoder = Decoder(num_layers, num_heads, d_model, d_k)
    self.positional_encoding = positional_encoding(d_model, dropout)

  def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor = None,
              tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
    """(batch, sequence, d_model) 형태의 텐서로 모델을 실행합니다."""
    # # PE
    # src = self.positional_encoding(src)
    # tgt = self.positional_encoding(tgt)
  
    # Encoder
    encoder_output = self.encoder(src, mask=src_mask)

    # Decoder
    output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
    return output
