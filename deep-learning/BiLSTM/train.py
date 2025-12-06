import torch
import torch.nn as nn
import torch.optim as optim
from seqeval.metrics import f1_score


def train_model(model, train_loader, device, num_epochs, learning_rate, tag2idx):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )
    
    idx2tag = {v: k for k, v in tag2idx.items()}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (words, tags, lengths, orig_indices) in enumerate(train_loader):
            words = words.to(device)
            tags = tags.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            
            tag_scores = model(words, lengths)
            
            batch_size, seq_len, tagset_size = tag_scores.size()
            
            tag_scores_flat = tag_scores.view(-1, tagset_size)
            tags_flat = tags.view(-1)
            
            loss = criterion(tag_scores_flat, tags_flat)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(tag_scores, dim=2)
            
            for i in range(batch_size):
                length = lengths[i].item()
                pred_tags = [idx2tag[predictions[i][j].item()] for j in range(length)]
                true_tags = [idx2tag[tags[i][j].item()] for j in range(length)]
                all_preds.append(pred_tags)
                all_labels.append(true_tags)
        
        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds)
        
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - F1: {train_f1:.4f}")


def predict(model, test_loader, device, idx2tag, pad_idx):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for words, _, lengths, global_indices in test_loader:
            words = words.to(device)
            lengths = lengths.to(device)
            
            tag_scores = model(words, lengths)
            
            pred_indices = torch.argmax(tag_scores, dim=2)
            
            batch_size = words.size(0)
            for i in range(batch_size):
                length = lengths[i].item()
                pred_tags = [idx2tag[pred_indices[i][j].item()] for j in range(length)]
                global_idx = global_indices[i]
                all_predictions.append((global_idx, pred_tags))
    
    all_predictions.sort(key=lambda x: x[0])
    
    predictions = [tags for _, tags in all_predictions]
    
    return predictions
