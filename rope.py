import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_k: int, base: float = 10000.0):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even d_k")

        # 차원 쌍마다 사용할 주파수
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        )

        # 학습 파라미터가 아닌 상수 텐서로 저장
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, heads, sequence, d_k)
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, heads, sequence, d_k)")
        if x.size(-1) != self.inv_freq.numel() * 2:
            raise ValueError("x의 마지막 차원이 초기화할 때 지정한 d_k와 달라요.")

        seq_len = x.size(-2)
        positions = torch.arange(
            seq_len, device=x.device, dtype=self.inv_freq.dtype
        )
        angles = positions[:, None] * self.inv_freq[None, :]

        cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]

        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated = torch.empty_like(x)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos

        return rotated