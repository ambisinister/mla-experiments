import torch

"""
RoPE taken from GPT-NeoX style, via llama in transformers.

https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247
"""

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def apply_rope_x(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

