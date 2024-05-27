import trimesh
import numpy as np
import plotly.graph_objects as go

def mesh_with_color(mesh_path):
    # with open(robot_data, 'r') as file:
    #     data = json.load(file)
    #
    # ee_pos = []
    # for i, item in enumerate(data['data']):
    #     T_ME = np.array(item["T_ME"])
    #     T_ME = T_ME.reshape(4, 4)
    #
    #     # get translation vector
    #     ee_pos.append(T_ME[:3, 3])
    #
    # ee_pos = np.array(ee_pos)
    mesh = trimesh.load(mesh_path)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    colors = np.array(mesh.visual.to_color().vertex_colors)

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            vertexcolor=colors
        ),
    #     go.Scatter3d(
    #         x=ee_pos[:, 0],
    #         y=ee_pos[:, 1],
    #         z=ee_pos[:, 2],
    #         mode='lines+markers',
    #         marker=dict(
    #             size=4,
    #             color=ee_pos[:, 2],
    #             colorscale='Viridis',
    #             opacity=0.8
    #         )
    #     ),
    #     go.Scatter3d(
    #         x=[ee_pos[0, 0]],
    #         y=[ee_pos[0, 1]],
    #         z=[ee_pos[0, 2]],
    #         mode='markers',
    #         marker=dict(
    #             size=8,
    #             color='red',
    #             opacity=1
    #         )
    #     )
    ])
    image_file = f'test.png'
    fig.write_image(image_file)

    fig.show()

if __name__ == "__main__":
    mesh_path = '../data/Paper/Paper_T1_Poke1/template_mesh/mesh-f00001.obj'
    mesh_with_color(mesh_path)