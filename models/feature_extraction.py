from torch import nn
import torch
from torchvision.models import resnet50, ResNet50_Weights

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


class ModifiedResNet50(nn.Module):
    def __init__(self, output_size=128):
        super(ModifiedResNet50, self).__init__()
        # Load the pretrained ResNet-50 model
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

        # Modify the FC layer to output size 128
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, output_size)
        )

        # Freeze the pretrained layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Only the parameters of the new layers will be trainable
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet50(x)