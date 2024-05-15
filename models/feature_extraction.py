from torch import nn
import torch

class ForceFeatures(nn.Module):
    def __init__(self):
        super(ForceFeatures, self).__init__()
        # Define a linear layer that maps 6 input features to 64 output features
        self.expand_features = nn.Linear(6, 64)

    def forward(self, x):
        # Apply the linear transformation
        x = self.expand_features(x)
        # Optionally, apply an activation function like ReLU to introduce non-linearity
        x = torch.relu(x)
        return x