from data import GPTTokenizedData
from model import Transformer
from train import train_model
from evaluation import perplexity
import torch

def main():
    # get dataloaders (data.py)
    tokenized = GPTTokenizedData()
    dataloaders = tokenized.dataloaders # all 3 dataloaders in a dictionary with keys 'train', 'test', 'val
    vocab_size = tokenized.vocab_size
    

    # instantiate model (model.py)

    model = Transformer(
        vocab_size=vocab_size,
        d=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=256,
        dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.device = device


    # train model (train.py)

    train_model(model, dataloaders['train'], val_dataloader=dataloaders['val'])
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    # evaluate perplexity for all three splits (evaluate.py)
    with torch.no_grad():
        train_perplexity, train_loss = perplexity(model, dataloaders['train'])
        val_perplexity, val_loss = perplexity(model, dataloaders['val'])
        test_perplexity, test_loss = perplexity(model, dataloaders['test'])
    print(f"Train Perplexity: {train_perplexity:.2f}, Loss: {train_loss:.2f}")
    print(f"Validation Perplexity: {val_perplexity:.2f}, Loss: {val_loss:.2f}")
    print(f"Test Perplexity: {test_perplexity:.2f}, Loss: {test_loss:.2f}")


if __name__ == "__main__":
    main()
