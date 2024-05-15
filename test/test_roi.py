import json
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pytorch3d.io import load_objs_as_meshes
import numpy as np
import torch

def plot_coordinate_system(fig, transformations, names, scale=20):
    for T, name in zip(transformations, names):
        # Origin
        origin = T[:3, 3]
        # Axes
        axes = scale * T[:3, :3]

        fig.add_trace(go.Scatter3d(x=[origin[0], origin[0] + axes[0, 0]],
                                   y=[origin[1], origin[1] + axes[1, 0]],
                                   z=[origin[2], origin[2] + axes[2, 0]],
                                   marker=dict(color='red'),
                                   line=dict(width=4),
                                   showlegend=False,
                                   name=f'{name}_X-axis'))

        fig.add_trace(go.Scatter3d(x=[origin[0], origin[0] + axes[0, 1]],
                                   y=[origin[1], origin[1] + axes[1, 1]],
                                   z=[origin[2], origin[2] + axes[2, 1]],
                                   marker=dict(color='green'),
                                   line=dict(width=4),
                                   showlegend=False,
                                   name=f'{name}_Y-axis'))

        fig.add_trace(go.Scatter3d(x=[origin[0], origin[0] + axes[0, 2]],
                                   y=[origin[1], origin[1] + axes[1, 2]],
                                   z=[origin[2], origin[2] + axes[2, 2]],
                                   marker=dict(color='blue'),
                                   line=dict(width=4),
                                   showlegend=False,
                                   name=f'{name}_Z-axis'))

def mesh_plotly(target_mesh, roi_mesh, names, ee_pos, transformation):
    target_mesh_verts = target_mesh.verts_packed().numpy()
    target_mesh_faces = target_mesh.faces_packed().numpy()
    roi_mesh_verts = roi_mesh.verts_packed().numpy()
    roi_mesh_faces = roi_mesh.faces_packed().numpy()
    ee_pos = ee_pos.numpy()

    fig = go.Figure(data=[
        go.Mesh3d(
            x=target_mesh_verts[:, 0],
            y=target_mesh_verts[:, 1],
            z=target_mesh_verts[:, 2],
            i=target_mesh_faces[:, 0],
            j=target_mesh_faces[:, 1],
            k=target_mesh_faces[:, 2],
            opacity=0.5,
            color='green',
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
    plot_coordinate_system(fig, [transformation], ["Test"])

    #fig.show()

def load_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            elif parts[0] == 'v':  # Vertex definition
                vertices.append([float(v) / 1000 for v in parts[1:4]])
            elif parts[0] == 'f':  # Face definition
                # Assumes that the OBJ file contains faces defined by vertex indices starting at 1
                faces.append([int(idx.split('/')[0]) - 1 for idx in parts[1:4]])

    return np.array(vertices), np.array(faces)

def visualize(verts, faces, ee_pos):

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.5,
            color='blue'
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
    # # Define the range for each axis
    # x_range = [-1, 1]  # Change these values to your desired range
    # y_range = [-1, 1]  # Change these values to your desired range
    # z_range = [-1, 1]  # Change these values to your desired range
    #
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(range=x_range),
    #         yaxis=dict(range=y_range),
    #         zaxis=dict(range=z_range),
    #         aspectmode = 'cube'  # Ensures same scale for x, y, z axes
    #     ),
    #     plot_bgcolor='white',  # Sets the background color to white
    #     legend=dict(
    #         title="Meshes",  # Optional: adds a title to the legend
    #         orientation="h",  # Horizontal legend orientation
    #         x=0.5,  # Center the legend horizontally
    #         xanchor="center",  # Anchor at the center for the x position
    #         y=1.05,  # Position the legend above the plot
    #         yanchor="bottom"  # Anchor at the bottom for the y position
    #     )
    # )

    fig.show()
def get_roi_meshes(meshes, target_position, rotation_matrix, threshold):
    selected_vertices, selected_faces = [], []
    face_indices_array = []
    for i, mesh in enumerate(meshes):
        pos = target_position[i].unsqueeze(0)
        distances_squared = torch.sum((mesh.verts_packed() - pos) ** 2, dim=1)

        # get vertices
        vertices_mask = distances_squared < threshold ** 2
        verts = mesh.verts_packed()[vertices_mask]
        vertices_indices = vertices_mask.nonzero(as_tuple=True)[0]

        # get directions
        directions = -(verts - pos)
        directions /= directions.norm(p=2, dim=1, keepdim=True)

        # filter for directions
        target_direction = rotation_matrix[i][:, 0]  # Adjust the index based on actual target
        print(target_direction)
        dot_products = torch.matmul(directions, target_direction)
        cosine_threshold = torch.cos(torch.deg2rad(torch.tensor(60.0)))
        similar_vectors = dot_products > cosine_threshold

        directions_mask = similar_vectors.nonzero(as_tuple=True)[0]

        vertices_indices = vertices_indices[directions_mask]

        # get faces
        faces = mesh.faces_packed()
        faces_mask = torch.zeros(faces.shape[0], dtype=torch.bool)
        for index in vertices_indices:
            faces_mask |= (faces == index).any(dim=1)

        face_indices = faces_mask.nonzero(as_tuple=True)[0]
        face_indices_array.append([face_indices])

    selected_meshes = meshes.submeshes(face_indices_array)

    return selected_meshes




if __name__ == '__main__':
    mesh_path = "../data/Couch_T1/triangle_meshes/mesh-f00166.obj"
    robot_data_path = "../data/Couch_T1/robot_data.json"
    with open(robot_data_path, "r") as f:
        json_data = json.load(f)

    for i, item in enumerate(json_data['data']):
        if int(item['frame']) == 166:
            T_ME = np.array(item['T_ME'])
            break

    ee_pos = torch.tensor(1000*T_ME[:3, 3]).unsqueeze(0)
    rotation_matrix = torch.tensor(T_ME[:3, :3]).unsqueeze(0)
    print(rotation_matrix)

    T_ME[:3, 3] = 1000*T_ME[:3, 3]

    meshes = load_objs_as_meshes([mesh_path])

    roi_meshes = get_roi_meshes(meshes, ee_pos, rotation_matrix, threshold=120)
    target_direction = rotation_matrix[0][:, 2]

    mesh_plotly(meshes, roi_meshes, ["Mesh", "ROI"], ee_pos[0], T_ME)

