import torch
import torch.nn as nn
from transformer import Transformer


def main():
  num_layers = 6
  num_heads = 8
  d_model = 512
  d_k = 64
  dropout = 0.1

  transformer = Transformer(num_layers, num_heads, d_model, d_k, dropout)

  loss = nn.MSELoss()

  src_data = torch.rand(10, 32, d_model)  # (seq_len, batch_size, d_model)
  tgt_data = torch.rand(20, 32, d_model)  # (seq_len, batch_size, d_model)

  output = transformer.forward(src_data, tgt_data)
  loss_value = loss(output, tgt_data)
  
  print(output)
  print(loss_value)

if __name__ == "__main__":
  main()