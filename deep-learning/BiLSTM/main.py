import sys
import torch
import random
import numpy as np

from dataset import SlotTaggingDataset, build_vocab, collate_fn
from model import LSTM
from train import train_model, predict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():    
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    
    word2idx, tag2idx = build_vocab(train_path)
    idx2tag = {v: k for k, v in tag2idx.items()}
    
    train_dataset = SlotTaggingDataset(train_path, word2idx, tag2idx, is_test=False)
    test_dataset = SlotTaggingDataset(test_path, word2idx, tag2idx, is_test=True)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print("Initializing model...")
    model = LSTM(
        vocab_size=len(word2idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        tagset_size=len(tag2idx),
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=word2idx['<PAD>']
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        tag2idx=tag2idx
    )
    
    predictions = predict(model, test_loader, device, idx2tag, word2idx['<PAD>'])
    
    with open(output_path, 'w') as f:
        f.write("ID,IOB Slot Tags\n")
        for idx, tags in enumerate(predictions):
            tag_str = ' '.join(tags)
            f.write(f"{idx + 1},{tag_str}\n")

if __name__ == "__main__":
    main()
