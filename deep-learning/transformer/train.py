"""
TODO write your training loop here.
Things to take care with:
    - make sure to use the correct loss function for the task
    - make sure that the targets are correct (each token should predict the next token in the sequence)
    - there should be no loss for padding tokens.
"""

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import time

def train_model(model, dataloader, save_path='best_model.pt', val_dataloader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)
    model.device = device

    loss_fn = CrossEntropyLoss(ignore_index = -100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01,)

    num_epochs = 10
    best_val_loss = float('inf')

    patience = 3
    patience_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        model.train()

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            attention_mask = attention_mask[:, :-1]

            targets = targets.masked_fill(attention_mask == 0, -100)

            optimizer.zero_grad()

            logits = model(inputs, attention_mask=attention_mask)
            logits = logits.view(-1, logits.size(-1)) # need to reshape based on vocab size

            targets = targets.view(-1)

            loss = loss_fn(logits, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

            elapsed = time.time() - start_time
            if elapsed > 18 * 60:
                torch.save(model.state_dict(), save_path)
                return model
        
        avg_loss = total_loss / num_batches

        if val_dataloader is not None:
            val_loss = evaluate_loss(model, val_dataloader, loss_fn, device)
            print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
    return model
        
def evaluate_loss(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            attention_mask = attention_mask[:, :-1]
            
            targets = targets.masked_fill(attention_mask == 0, -100)
            
            logits = model(inputs, attention_mask=attention_mask)
            
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches
