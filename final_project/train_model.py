import torch
import numpy as np
from modeling.gpt import GPTModel
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
import time

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

# Helper function to plot a loss curve every few batches
def plot_loss_curve(x, y):
    plt.plot(x, y)
    plt.title("LLM Training Loss w/ MHA + Decoupled RoPE")
    plt.xlabel("tokens")
    plt.ylabel("cross entropy loss")
    plt.savefig("./figures/mha_decoupled_rope_training_curve.png")

def train():
    # using nvidia rtx 3090
    # roughly gpt-2-medium
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = GPTModel(d_model=1024, n_heads=16, layers=24, vocab_size=10000,
    #                  max_seq_len=1024, use_mla=True, use_mqa=False)

    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000,
                     max_seq_len=1024, use_rope=True, use_decoupled=True)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    model = model.to(device)

    # gradient scaling and accumulation 
    scaler = GradScaler()
    acc_steps = 4

    batch_size = 12 # should fit on 3090, might take a while
    # lr and betas for adamW from gpt-2-medium
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999)) 
    # determine cosine schedule based on roughly total steps, ~100m token dataset
    # use 1% of the steps for warmup
    total_steps = 1e8 / (batch_size * acc_steps)
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, int(total_steps * 0.01))
    loss_fn = torch.nn.CrossEntropyLoss()

    # loading the data
    with open('./data/packed_data.npy', 'rb') as f:
        data = np.load(f)

    dataset = TensorDataset(torch.from_numpy(data[:, :-1]), torch.from_numpy(data[:, 1:]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    #logging
    total_tokens = 0
    train_losses_y = []
    train_losses_x = []

    # epochs = 1
    for i, (dat, targ) in enumerate(dataloader):
        dat, targ = dat.to(device, non_blocking=True), targ.to(device, non_blocking=True)
        print(f"{i}/{len(dataloader)}")

        # https://pytorch.org/docs/stable/amp.html
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out, _ = model(dat)
            out = out.permute(0, 2, 1)
            loss = loss_fn(out, targ)
            loss = loss / acc_steps

        scaler.scale(loss).backward()

        if (i+1) % acc_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt.zero_grad(set_to_none=True)

        # logging total tokens is just S * batch size
        total_tokens += dat.numel()

        # log every 10 effective batches
        if (i + 1) % (10 * acc_steps) == 0:
            train_losses_x.append(total_tokens)
            train_losses_y.append(loss.item())
            print(f"{i}/{len(dataloader)}", loss.item())
            plot_loss_curve(train_losses_x, train_losses_y)

    # save model weights
    torch.save(model.state_dict(), "./weights/model_weights.pt")

if __name__ == "__main__":
    train()
