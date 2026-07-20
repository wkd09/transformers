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

  src_data = torch.rand(32, 10, d_model)  # (batch_size, seq_len, d_model)
  tgt_data = torch.rand(32, 20, d_model)  # (batch_size, seq_len, d_model)

  output = transformer.forward(src_data, tgt_data)
  loss_value = loss(output, tgt_data)
  
  loss_value.backward()
  parameter_count = sum(parameter.numel() for parameter in transformer.parameters())

  print(f"output shape: {tuple(output.shape)}")
  print(f"loss: {loss_value.item():.4f}")
  print(f"trainable parameters: {parameter_count:,}")

if __name__ == "__main__":
  main()
