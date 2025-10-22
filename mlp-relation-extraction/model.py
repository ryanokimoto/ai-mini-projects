"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the model architecture that you need to implement for HW1.
You should complete the BoWClassifier class by implementing the forward method
and any other necessary components.
"""

import torch
from torch import nn

class BoWClassifier(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = 0.3
        self.hidden_sizes = [128, 256]
        self.use_layer_norm = False

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate([128]):
            layers.append(nn.Linear(prev_size, hidden_size))

            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            layers.append(nn.ReLU())

            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_labels))
        self.network = nn.Sequential(*layers)

        
        
    def forward(self, x):
        logits = self.network(x)
        return torch.sigmoid(logits)
    
    
def get_best_model(input_size, num_labels):
    # return a newly instantiated model that your best weights will be loaded into
    # the model returned by this function must exactly match the architecture that the saved weights expect
    return BoWClassifier(input_size=input_size, num_labels=num_labels)

def predict(model_output):
    """
    Converts model output to class predictions.
    Args:
        model_output: Output from model.forward(x)
    Returns:
        predictions: Tensor of predicted class labels
    """
    return model_output > 0.5

