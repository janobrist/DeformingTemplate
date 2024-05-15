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

def mesh_plotly(target_mesh, roi_mesh, names, ee_pos):
    target_mesh_verts = target_mesh.verts_packed().detach().cpu().numpy()
    target_mesh_faces = target_mesh.faces_packed().detach().cpu().numpy()
    roi_mesh_verts = roi_mesh.verts_packed().detach().cpu().numpy()
    roi_mesh_faces = roi_mesh.faces_packed().detach().cpu().numpy()
    ee_pos = ee_pos.detach().cpu().numpy()

    fig = go.Figure(data=[
        go.Mesh3d(
            x=target_mesh_verts[:, 0],
            y=target_mesh_verts[:, 1],
            z=target_mesh_verts[:, 2],
            i=target_mesh_faces[:, 0],
            j=target_mesh_faces[:, 1],
            k=target_mesh_faces[:, 2],
            opacity=0.5,
            color='blue',
            name=names[0],
        ),
        go.Mesh3d(
            x=roi_mesh_verts[:, 0],
            y=roi_mesh_verts[:, 1],
            z=roi_mesh_verts[:, 2],
            i=roi_mesh_faces[:, 0],
            j=roi_mesh_faces[:, 1],
            k=roi_mesh_faces[:, 2],
            opacity=0.5,
            color='red',
            name=names[1],
        ),
        go.Scatter3d(
            x=[ee_pos[0]],
            y=[ee_pos[1]],
            z=[ee_pos[2]],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                opacity=1
            )
        )
    ])
    # Define the range for each axis
    x_range = [-1, 1]  # Change these values to your desired range
    y_range = [-1, 1]  # Change these values to your desired range
    z_range = [-1, 1]  # Change these values to your desired range

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode = 'cube'  # Ensures same scale for x, y, z axes
        ),
        plot_bgcolor='white',  # Sets the background color to white
        legend=dict(
            title="Meshes",  # Optional: adds a title to the legend
            orientation="h",  # Horizontal legend orientation
            x=0.5,  # Center the legend horizontally
            xanchor="center",  # Anchor at the center for the x position
            y=1.05,  # Position the legend above the plot
            yanchor="bottom"  # Anchor at the bottom for the y position
        )
    )

    return fig
