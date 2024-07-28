import torch
import numpy as np
from modeling.gpt import GPTModel

def load_model(model_path, device, use_mla=False, use_mqa=False, use_rope=False):
    # model = GPTModel(d_model=1024, n_heads=16, layers=24, vocab_size=10000,
    #                  max_seq_len=1024, use_mla=use_mla, use_mqa=use_mqa)
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000,
                     max_seq_len=1024, use_mla=use_mla, use_mqa=use_mqa,
                     use_rope=use_rope, use_decoupled=True)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def calculate_perplexity(model, data, batch_size, device):
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(data) - batch_size, batch_size):            
            print(f"{i} / {len(data)}")
            batch = data[i:i + batch_size]
            inputs = torch.tensor(batch[:, :-1]).to(device)
            targets = torch.tensor(batch[:, 1:]).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)),
                                                         targets.view(-1), reduction='sum')
                
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model_path = "./weights/reference_model.pt"
    #model_path = "./weights/35m_model.pt"
    #model_path = "./weights/mqa_model.pt"
    model_path = "./weights/model_weights.pt" 
    use_mla = False
    use_mqa = False
    use_rope = True
    data_path = "./data/packed_data.npy"
    batch_size = 16

    # Load the model
    model = load_model(model_path, device,
                       use_mla=use_mla, use_mqa=use_mqa,
                       use_rope=use_rope)

    # Load the data
    with open(data_path, 'rb') as f:
        data = np.load(f)

    # Calculate perplexity
    perplexity = calculate_perplexity(model, data, batch_size, device)
    print(f"Training Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()
