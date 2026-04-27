import torch.nn as nn
import torch
from Attention import attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads:int, d_model:int, d_k:int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k

        #선형 변환으로 다양한 attention 관점을 갖게 한다.
        self.q_linear = nn.Linear(d_model, num_heads * d_k)
        self.k_linear = nn.Linear(d_model, num_heads * d_k)
        self.v_linear = nn.Linear(d_model, num_heads * d_k)
        self.out_linear = nn.Linear(num_heads * d_k, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        batch_size = q.size(0)

        #attention 헤드 분리한다(transpose로 차원 재배치)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        attn_output = attention(q, k, v, mask)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.out_linear(attn_output)

        return output