import torch

# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")
c_float32 = torch.rand((8, 8), device="cuda")
d_float32 = torch.rand((8, 8), device="cuda")

with torch.autocast(device_type="cuda"):
    # torch.mm is on autocast's list of ops that should run in float16.
    # Inputs are float32, but the op runs in float16 and produces float16 output.
    # No manual casts are required.
    e_float16 = torch.mm(a_float32, b_float32)
    # Also handles mixed input types
    f_float16 = torch.mm(d_float32, e_float16)

# After exiting autocast, calls f_float16.float() to use with d_float32
g_float32 = torch.mm(d_float32, f_float16.float())

print(g_float32, e_float16)
