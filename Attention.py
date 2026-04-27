import torch
import math

def attention(q, k, v, mask=None):
    d_k = q.size(-1)

    # 쿼리랑 키(전치한다)를 내적하고 스케일링
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = torch.softmax(scores, dim=-1)
    output = attn @ v

    return output