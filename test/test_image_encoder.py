from torchvision.models import vgg19, VGG19_Weights, resnet50, ResNet50_Weights, vit_l_32, ViT_L_32_Weights, mobilenet_v3_large
from PIL import Image
from torchvision import transforms
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import trimesh
import open3d as o3d
def compare_features(template_images, target_images):
    image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
    template_features = image_encoder.forward(template_images)
    target_features = image_encoder.forward(target_images)
    #loss = F.mse_loss(template_features, target_features)
    # Flatten the feature maps
    features_img1_flat = template_features.view(template_images.size(0), -1)
    features_img2_flat = target_features.view(target_features.size(0), -1)

    # Calculate the cosine similarity
    loss = F.cosine_similarity(features_img1_flat, features_img2_flat)
    #print(loss)
    #visualize_features(template_features, 'Template Image Features')
    #visualize_features(target_features, 'Target Image Features')

    print(loss)

# Function to visualize feature maps
def visualize_features(features, title):
    features = features.detach().numpy()
    plt.figure(figsize=(10, 6))
    sns.heatmap(features, annot=False, cmap='viridis')
    plt.title(title)
    plt.xlabel('Feature Dimension')
    plt.ylabel('Image Index')
    plt.show()


if __name__ == "__main__":

    image_root = "../data/Couch_T1/images"
    transformation = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256
        transforms.CenterCrop(224),  # Crop a 224x224 patch from the center
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalize with ImageNet's mean and std
    ])
    cameras = ["0005", "0029", "0068", "0089"]
    template_images = []
    target_images = []
    for camera in cameras:
        image_path = os.path.join(image_root, camera, f"{1:05d}.png")
        image = Image.open(image_path)
        image = transformation(image)
        template_images.append(image)

        image_path = os.path.join(image_root, camera, f"{22:05d}.png")
        image = Image.open(image_path)
        image = transformation(image)
        target_images.append(image)

    template_images = torch.stack(template_images)
    target_images = torch.stack(target_images)

    compare_features(template_images, target_images)

