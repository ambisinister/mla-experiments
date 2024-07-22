import torch
from matplotlib import pyplot as plt
import time

from ..modeling.gpt import GPTModel
from ..hftokenizer import HFTokenizer

def plot_100_out(xs, ys, labs):
    plt.title("Time to generate 100 tokens")
    plt.xlabel("Input Prompt Size")
    plt.ylabel("Time")
    for x, y, z in zip(xs, ys, labs):
        plt.plot(x, y, label=z)
    plt.legend()
    plt.savefig("./figures/inference_100_out.png")
    plt.clf()

def plot_100_in(xs, ys, labs):
    plt.title("Time to generate w/ 100 tokens input")
    plt.xlabel("Output Token Count")
    plt.ylabel("Time")
    for x, y, z in zip(xs, ys, labs):
        plt.plot(x, y, label=z)
    plt.legend()
    plt.savefig("./figures/inference_100_in.png")
    plt.clf()
    
def load_model(model_path, device, model_kwargs=None):
    if model_kwargs == None:
        model_kwargs = {
            "d_model": 512,
            "n_heads": 16,
            "layers": 8,
            "vocab_size": 10000,
            "max_seq_len": 256,
            "use_mla": False,
            "cache_compress": True
        }
    model = GPTModel(**model_kwargs)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, num_tokens_to_generate, device):
    model.eval()
    encoded_prompt = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded_prompt).unsqueeze(0).to(device)
    
    generated_tokens = encoded_prompt
    kv_cache = None
    past_length = 0
    
    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            if kv_cache is None:
                # for first iteration, process whole prompt
                logits, kv_cache = model(input_ids)
                next_token_logits = logits[:, -1, :]
                past_length += input_ids.size()[-1]
            else:                
                # afterwards, just do last token
                logits, kv_cache = model(input_ids[:, -1:], kv_cache, past_length=past_length)
                next_token_logits = logits[:, 0, :]
                past_length += 1

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(generated_tokens)

def experiment_out(model, tokenizer, device, prompt_len):
    prompt = " the" * prompt_len

    tick = time.time()
    out = generate_text(model, tokenizer, prompt, 100, device)
    tock = time.time()
    return tock - tick

def experiment_in(model, tokenizer, device, output_len):
    prompt = " the" * 100

    tick = time.time()
    out = generate_text(model, tokenizer, prompt, output_len, device)
    tock = time.time()
    return tock - tick

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_ref = ["./weights/reference_model.pt"]
    models_mla = ["./weights/31m_model.pt", "./weights/35m_model.pt"]

    # Load the tokenizer (common to all)
    tokenizer = HFTokenizer()
    tokenizer.load()

    xs, ys, labs = [], [], []

    # Run experiments for each model
    for experiment in ["in", "out"]:
        for model_str in [*models_ref, *models_mla]:
            print(model_str)
            model_kwargs = {
                "d_model": 512,
                "n_heads": 16,
                "layers": 8,
                "vocab_size": 10000,
                "max_seq_len": 256,
                "use_mla": False,
                "cache_compress": False
            }
        
            if model_str in models_ref:
                model_kwargs["use_mla"] = False
                exps = 1
            if model_str in models_mla:
                model_kwargs["use_mla"] = True
                exps = 2
                if "35m" in model_str:
                    model_kwargs["layers"] = 9

            for e in range(exps):
                print(e)
                if e == 1:
                    model_kwargs["cache_compress"] = True
                model = load_model(model_str, device, model_kwargs)

                this_x, this_y = [], []

                for tokens in range(10, 100):
                    this_x.append(tokens)
            
                    if experiment == "out":
                        this_y.append(experiment_out(model, tokenizer, device, tokens))
                    else:
                        this_y.append(experiment_in(model, tokenizer, device, tokens))

                xs.append(this_x)
                ys.append(this_y)
                labs.append(model_str + "_compress_kv" if e == 1 else model_str)

        if experiment == "out":
            plot_100_out(xs, ys, labs)
        else:
            plot_100_in(xs, ys, labs)
    
if __name__ == "__main__":
    main()
