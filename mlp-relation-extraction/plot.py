"""
Script to plot training and validation loss curves.
"""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(history, save_path='loss_curves.png'):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot losses
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
    
    # Formatting
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (BCE)', fontsize=14)
    plt.title('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add markers for best validation loss
    best_val_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    plt.axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.5, 
                label=f'Best Val Loss (Epoch {best_val_epoch})')
    plt.scatter([best_val_epoch], [best_val_loss], color='green', s=100, zorder=5)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")
    plt.show()


def plot_f1_curves(history, save_path='f1_curves.png'):
    """
    Plot training and validation F1 score curves.
    
    Args:
        history: Dictionary containing 'train_f1' and 'val_f1' lists
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['train_f1']) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot F1 scores
    plt.plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2, marker='o')
    plt.plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2, marker='s')
    
    # Formatting
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Weighted F1 Score', fontsize=14)
    plt.title('Training and Validation F1 Score Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add markers for best validation F1
    best_val_epoch = np.argmax(history['val_f1']) + 1
    best_val_f1 = max(history['val_f1'])
    plt.axvline(x=best_val_epoch, color='green', linestyle='--', alpha=0.5,
                label=f'Best Val F1 (Epoch {best_val_epoch})')
    plt.scatter([best_val_epoch], [best_val_f1], color='green', s=100, zorder=5)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"F1 curve saved to {save_path}")
    plt.show()


def plot_combined_curves(history, save_path='combined_curves.png'):
    """
    Plot both loss and F1 curves in a single figure with two subplots.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss subplot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (BCE)', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Mark best validation loss
    best_val_loss_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_val_loss_epoch, color='green', linestyle='--', alpha=0.5)
    ax1.scatter([best_val_loss_epoch], [best_val_loss], color='green', s=100, zorder=5)
    
    # F1 subplot
    ax2.plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2, marker='o')
    ax2.plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Weighted F1 Score', fontsize=12)
    ax2.set_title('F1 Score Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Mark best validation F1
    best_val_f1_epoch = np.argmax(history['val_f1']) + 1
    best_val_f1 = max(history['val_f1'])
    ax2.axvline(x=best_val_f1_epoch, color='green', linestyle='--', alpha=0.5)
    ax2.scatter([best_val_f1_epoch], [best_val_f1], color='green', s=100, zorder=5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined curves saved to {save_path}")
    plt.show()


def main():
    """Load history and create plots."""
    # Load training history
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    print(f"Loaded training history with {len(history['train_loss'])} epochs")
    
    # Create individual plots
    plot_loss_curves(history, 'loss_curves.png')
    plot_f1_curves(history, 'f1_curves.png')
    
    # Create combined plot
    plot_combined_curves(history, 'combined_curves.png')
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Best Training Loss:   {min(history['train_loss']):.4f} (Epoch {np.argmin(history['train_loss'])+1})")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss'])+1})")
    print(f"Best Training F1:     {max(history['train_f1']):.4f} (Epoch {np.argmax(history['train_f1'])+1})")
    print(f"Best Validation F1:   {max(history['val_f1']):.4f} (Epoch {np.argmax(history['val_f1'])+1})")
    print(f"Final Training Loss:  {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Training F1:    {history['train_f1'][-1]:.4f}")
    print(f"Final Validation F1:  {history['val_f1'][-1]:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()