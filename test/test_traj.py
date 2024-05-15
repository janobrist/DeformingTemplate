import json
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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

    return fig



if __name__ == '__main__':
    mesh_path = "../data/Couch_T1/triangle_meshes/mesh-f00166.obj"
    robot_data_path = "../data/Couch_T1/robot_data.json"
    with open(robot_data_path, "r") as f:
        json_data = json.load(f)
    verts, faces = load_obj(mesh_path)

    for i, item in enumerate(json_data['data']):
        if int(item['frame']) == 166:
            T_ME = np.array(item['T_ME'])
            break

    ee_pos = T_ME[:3, 3]
    print(ee_pos)

    fig = visualize(verts, faces, ee_pos)
    fig.show()
