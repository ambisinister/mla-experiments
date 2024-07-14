import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt

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
    plt.title("LLM Training Loss w/ MLA + add back saved parameters")
    plt.xlabel("tokens")
    plt.ylabel("cross entropy loss")
    plt.savefig("./training_curve.png")

def train():
    # using nvidia rtx 3090, all left the same as originally
    # 35M parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000, max_seq_len=256)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    model = model.to(device)

    batch_size = 128 # fairly large batch size since we have the memory
    # lr and betas for adamW from gpt-2-small
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95)) 
    # determine cosine schedule based on roughly total steps, ~100m token dataset
    # use 1% of the steps for warmup
    total_steps = 1e8 / batch_size 
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, int(total_steps * 0.01))
    loss_fn = torch.nn.CrossEntropyLoss()

    # loading the data
    with open('packed_data.npy', 'rb') as f:
        data = np.load(f)

    #logging
    total_tokens = 0
    train_losses_y = []
    train_losses_x = []

    # epochs = 1
    for i in range(0, len(data) - batch_size, batch_size):
        opt.zero_grad()
        # batch the data
        batch = data[i:i + batch_size]

        # offsets, converting to tensor, and sending to gpu
        dat = torch.tensor(batch[:, :-1]).to(device)
        targ = torch.tensor(batch[:, 1:]).to(device)

        # output is the wrong size by default so we have to permute the dims
        out = model(dat)
        out = out.permute(0, 2, 1)
        loss = loss_fn(out, targ)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        scheduler.step()

        # logging total tokens is just S * batch size
        total_tokens += np.prod([*np.shape(batch)])

        # log every 10 batches, or every 327,680 tokens
        if i % 10 == 9:
            train_losses_x.append(total_tokens)
            train_losses_y.append(loss.item())
            print(i, loss.item())
            plot_loss_curve(train_losses_x, train_losses_y)

    # save model weights
    torch.save(model.state_dict(), "./model_weights.pt")

if __name__ == "__main__":
    train()
