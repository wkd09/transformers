import torch
import torch.nn as nn

class positional_encoding(nn.Module):
  def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
    super(positional_encoding, self).__init__()
    
    # pos 정의 벡터
    pos = torch.arange(0, max_len).reshape(max_len, 1).float()
    # d_model 차원 정의
    den = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float32) * torch.log(torch.tensor(10000.0)) / d_model)

    # pos_embedding 정의
    pos_embedding = torch.zeros((max_len, d_model), dtype=torch.float32)
    # 짝수 인덱스 sin, 홀수 인덱스 cos
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    # shape (max_len, d_model) -> (max_len, 1, d_model)
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.dropout = torch.nn.Dropout(dropout)
    self.register_buffer('pos_embedding', pos_embedding)

  def forward(self, token_embedding: torch.Tensor):
    return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])