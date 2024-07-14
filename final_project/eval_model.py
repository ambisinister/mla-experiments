import torch
import numpy as np
from gpt import GPTModel

def load_model(model_path, device):
    model = GPTModel(d_model=512, n_heads=16, layers=9, vocab_size=10000, max_seq_len=256)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def calculate_perplexity(model, data, batch_size, device):
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(data) - batch_size, batch_size):
            batch = data[i:i + batch_size]
            inputs = torch.tensor(batch[:, :-1]).to(device)
            targets = torch.tensor(batch[:, 1:]).to(device)
            
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)),
                                                     targets.view(-1), reduction='sum')
            
            total_loss += loss.item()
            total_tokens += np.prod(targets.size())
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model_weights.pt"
    data_path = "packed_data.npy"
    batch_size = 128

    # Load the model
    model = load_model(model_path, device)

    # Load the data
    with open(data_path, 'rb') as f:
        data = np.load(f)

    # Calculate perplexity
    perplexity = calculate_perplexity(model, data, batch_size, device)
    print(f"Training Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()
