import torch
from modeling.gpt import GPTModel
from hftokenizer import HFTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device, use_mla=False, use_mqa=False, use_rope=False):
    # model = GPTModel(d_model=1024, n_heads=16, layers=24, vocab_size=10000,
    #                  max_seq_len=1024, use_mla=use_mla, use_mqa=use_mqa)
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000,
                     max_seq_len=1024, use_mla=use_mla, use_mqa=use_mqa,
                     use_rope=use_rope, use_decoupled=True)
    #model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def calculate_cache_size(kv_cache):
    # MHA / MQA
    if isinstance(kv_cache[0], tuple):
        total_params = sum(k.numel() + v.numel() for k, v in kv_cache)
        if len(kv_cache[0][0].size()) == 3:
            num_tokens = kv_cache[0][0].size(1)
        else:
            num_tokens = kv_cache[0][0].size(2)
    # MLA
    else:  
        total_params = sum(tensor.numel() for tensor in kv_cache)
        num_tokens = kv_cache[0].size(1)
    
    params_per_token = total_params / num_tokens
    return params_per_token

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
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, kv_cache = model(input_ids)

                next_token_logits = logits[:, -1, :]
                past_length += input_ids.size()[-1]
            else:                
                # afterwards, just do last token
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, kv_cache = model(input_ids[:, -1:], kv_cache,
                                             past_length=past_length)
                next_token_logits = logits[:, 0, :]
                past_length += 1

            # Debugging: Compare with full forward pass
            with torch.autocast(device_type="cuda"):
                full_logits, _ = model(input_ids)
            diff = torch.abs(full_logits[:, -1, :] - logits[:, -1, :]).max()
            print(f"Max difference at step {past_length}: {diff.item()}")
                
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            print(f"Params per token: {calculate_cache_size(kv_cache)}")
    
    return tokenizer.decode(generated_tokens)
                                    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu") # this is not the same???????
    #model_path = "./weights/reference_model.pt"
    model_path = "./weights/model_weights.pt"
    use_mla=False
    use_mqa=False
    use_rope=True
    #model_path = "./weights/31m_model.pt"
    #use_mla=True
    #use_mqa=True
    #model_path = "./weights/mqa_model.pt"
    #use_mla=False
    #use_mqa=True
    
    prompt = "There once was a monster."
    num_tokens_to_generate = 20

    # Load the model
    model = load_model(model_path, device, use_mla=use_mla, use_mqa=use_mqa, use_rope=use_rope)

    # Load the tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, num_tokens_to_generate, device)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
