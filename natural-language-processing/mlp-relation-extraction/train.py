"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the training loop that you need to implement for HW1.
You should complete the train_model function by implementing the training logic
including optimizer setup, loss function, training loop, and model saving.

TODO: Implement the training loop in the train_model function
"""

# define your training loop here
import torch
from eval import evaluate_metrics

def train_model(model, predict_fn, train_loader, val_loader, device, save_path='best_model.pt'):
    model = model.to(device)
    
    model.train()
    
    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    best_f1_weighted = 0.0
    num_epochs = 20
    epochs_without_improvement = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float())

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                total_val_loss += loss.item()
                num_val_batches += 1
        avg_val_loss = total_val_loss / num_val_batches
        train_f1, _, _, _, _ = evaluate_metrics(model, train_loader, predict_fn, device)
        val_f1, _, _, _, _ = evaluate_metrics(model, val_loader, predict_fn, device)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Weighted F1: {train_f1:.4f}, Val Loss: {avg_val_loss:.4f}, Val Weighted F1: {val_f1:.4f}")

        if val_f1 > best_f1_weighted:
            best_f1_weighted = val_f1
            torch.save(model.state_dict(), save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 3:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(save_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    f1_weighted, precision, recall, f1, support = evaluate_metrics(model, val_loader, predict_fn, device)
    print(f"*** Best (weighted) F1: {f1_weighted} ***")
    print(f'*** Best model weights saved at {save_path} ***')
    
    return model, history
    