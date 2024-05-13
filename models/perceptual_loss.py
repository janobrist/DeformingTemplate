import torch
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
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
