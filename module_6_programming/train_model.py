import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt


# since we didn't really cover how to do this in lecture-
# this creates a learning rate schedule for you. Refer to the 
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step. 
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

# ===========================================================================

'''
Complete the following method which trains a GPT model and saves a loss curve.
'''
def plot_loss_curve(x, y):
    plt.plot(x, y)
    plt.title("LLM Training Loss")
    plt.xlabel("tokens")
    plt.ylabel("cross entropy loss")
    plt.savefig("./training_curve.png")

def train():
    device = torch.device("cuda")    
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000, max_seq_len=256)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    model = model.to(device)

    batch_size = 128
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95)) # lr of gpt-2-small
    total_steps = 1e8 / batch_size
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, int(total_steps * 0.01))
    loss_fn = torch.nn.CrossEntropyLoss()

    with open('packed_data.npy', 'rb') as f:
        data = np.load(f)

    # We have to batch the data
    full_batches = np.shape(data)[0] // batch_size
    batched_data = data[:full_batches * batch_size].reshape(-1, batch_size, np.shape(data)[-1])

    #logging
    total_tokens = 0
    train_losses_y = []
    train_losses_x = []

    # epochs = 1
    for i,batch in enumerate(batched_data):
        opt.zero_grad()

        dat = batch[:, :-1]
        dat = torch.tensor(dat).to(device)
        targ = batch[:, 1:]
        targ = torch.tensor(targ).to(device)

        out = model(dat)
        out = out.permute(0, 2, 1)
        loss = loss_fn(out, targ)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        scheduler.step()

        # logging total tokens is just S * batch size
        total_tokens += np.prod([*np.shape(batch)])

        if i % 10 == 9:
            train_losses_x.append(total_tokens)
            train_losses_y.append(loss.item())
            print(i, loss.item())
            plot_loss_curve(train_losses_x, train_losses_y)

    # save model weights if you want
    torch.save(model.state_dict(), "./model_weights.pt")


if __name__ == "__main__":
    train()
