import torch
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
from torch import nn
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        for x in range(14):  # Up to ReLU3_3
            self.slice1.add_module(str(x), vgg[x])
        for param in self.slice1.parameters():
            param.requires_grad = False

    def forward(self, predicted, target):
        pred_features = self.slice1(predicted)
        target_features = self.slice1(target)
        return F.l1_loss(pred_features, target_features)

class MaskedPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedPerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.features = torch.nn.Sequential()
        # Using up to ReLU3_3 as in your original setup
        for x in range(14):
            self.features.add_module(str(x), vgg[x])
        # Freeze the VGG parameters as we do not want to train them
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, predicted, target, mask):
        # Apply the mask by element-wise multiplication
        pred_masked = predicted * mask
        target_masked = target * mask
        # Extract features
        pred_features = self.features(pred_masked)
        target_features = self.features(target_masked)
        # Calculate L1 Loss between the features of the masked regions
        return F.l1_loss(pred_features, target_features)

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
