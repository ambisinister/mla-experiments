import torch
import math

q_heads = torch.rand((1, 16, 1, 32), device="cuda")
k_heads = torch.rand((1, 16, 28, 32), device="cuda")
v_heads = torch.rand((1, 16, 28, 32), device="cuda")

S = 1
S_full = 28
past_length = 27

# method B

mask = torch.ones((S,S_full), device="cuda")
mask = torch.tril(mask, diagonal=past_length)
mask = mask[None, None, :, :]

k_heads_t = torch.transpose(k_heads, -2, -1)
qkt = torch.matmul(q_heads, k_heads_t) / math.sqrt(float(32))
qkt = qkt * mask
qkt[qkt == 0] = float('inf')
attn = torch.nn.functional.softmax(qkt, dim=-1)
x = torch.matmul(attn, v_heads)

b = x
b = b.transpose(1,2).reshape(1, 1, 16*32)

# method A

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to("cuda")
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

# perform some conversion steps here
sq_mask = mask[0, 0, :, :]
# should be the same as print(x)
a = scaled_dot_product_attention(q_heads, k_heads, v_heads, attn_mask=sq_mask)
a = a.transpose(1,2).reshape(1, S, 16*32)

print(a-b)
