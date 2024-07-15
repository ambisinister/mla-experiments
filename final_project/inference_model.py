import torch
from gpt import GPTModel
from hftokenizer import HFTokenizer

def load_model(model_path, device, use_mla=False):
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000, max_seq_len=256, use_mla=use_mla)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def calculate_cache_size(kv_cache):
    # MHA 
    if isinstance(kv_cache[0], tuple):
        total_params = sum(k.numel() + v.numel() for k, v in kv_cache)
        num_tokens = kv_cache[0][0].size(1)
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
    
    with torch.no_grad():
        for _ in range(num_tokens_to_generate):
            #if kv_cache is None:
            if True:
                # for first iteration, process whole prompt
                logits, kv_cache = model(input_ids)
                next_token_logits = logits[:, -1, :]
            else:
                # afterwards, just do last token
                logits, kv_cache = model(input_ids[:, -1:], kv_cache)
                next_token_logits = logits[:, 0, :]
                print(f"Params per token: {calculate_cache_size(kv_cache)}")
            
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return tokenizer.decode(generated_tokens)
                                    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model_path = "./weights/reference_model.pt"
    #use_mla=False
    model_path = "./weights/31m_model.pt"
    use_mla=True
    
    prompt = "There once was a monster."
    num_tokens_to_generate = 20

    # Load the model
    model = load_model(model_path, device, use_mla=use_mla)

    # Load the tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, num_tokens_to_generate, device)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
