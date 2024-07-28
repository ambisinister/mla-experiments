# This has been helpful for sanity checking

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

from ..modeling.gpt import GPTModel

def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler


def setup_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(d_model=512, n_heads=16, layers=8,
                     vocab_size=10000, max_seq_len=1024,
                     use_mla=False, use_mqa=False, use_rope=False)
    model = model.to(device)

    scaler = GradScaler()
    acc_steps = 4

    batch_size = 12
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999)) 
    total_steps = 1e8 / (batch_size * acc_steps)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, int(total_steps * 0.01))
    loss_fn = torch.nn.CrossEntropyLoss()

    # loading the data
    with open('./data/packed_data.npy', 'rb') as f:
        data = np.load(f)

    dataset = TensorDataset(torch.from_numpy(data[:, :-1]), torch.from_numpy(data[:, 1:]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return model, opt, dataloader, loss_fn, scaler, device

def train_step(model, data, target, optimizer, loss_fn, scaler, device):
    data, target = data.to(device), target.to(device)
    
    with torch.cuda.amp.autocast():
        output, _ = model(data)
        loss = loss_fn(output.view(-1, output.size(-1)), target.view(-1))
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

def profile_training(num_steps=500):
    model, optimizer, dataloader, loss_fn, scaler, device = setup_training()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i, (data, target) in enumerate(dataloader):
            if i >= num_steps:
                break
            with record_function("training_step"):
                train_step(model, data, target, optimizer, loss_fn, scaler, device)
            prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print("\nCUDA Memory Usage:")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

if __name__ == "__main__":
    profile_training()
    print("Profiling complete.")
