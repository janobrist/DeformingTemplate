import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def inverse_normalize(tensor, mean, std):
    """
    Reverse the normalization of an image tensor.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor.clone()  # Clone the tensor to avoid changing it in place
    tensor.mul_(std).add_(mean)  # Reverse the normalization
    return tensor

def show_image(tensor, transform_back):
    """
    Display an image tensor using Matplotlib.
    """
    if transform_back:
        tensor = inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.cpu().numpy().transpose(1, 2, 0)  # Convert from PyTorch (C, H, W) to Matplotlib (H, W, C) format
    tensor = tensor.clip(0, 1)  # Ensure the image data is within [0, 1]

    plt.imshow(tensor)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def mesh_plotly(vertices, faces):
    vertices = vertices.cpu().numpy()
    faces = faces.cpu().numpy()

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.5,
            color='blue'
        )
    ])
    return fig
