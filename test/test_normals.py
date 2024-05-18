import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
import numpy as np
import plotly.graph_objects as go
from pytorch3d.ops import sample_points_from_meshes
import torch.nn.functional as F

def mesh_plotly(target_mesh, template_mesh, names, tar_pos, temp_pos):
    target_mesh_verts = target_mesh.verts_packed().numpy()
    target_mesh_faces = target_mesh.faces_packed().numpy()
    roi_mesh_verts = template_mesh.verts_packed().numpy()
    roi_mesh_faces = template_mesh.faces_packed().numpy()
    tar_pos = tar_pos.numpy()
    temp_pos = temp_pos.numpy()

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
            color='blue',
            name=names[1],
        ),
        go.Scatter3d(
            x=[tar_pos[0]],
            y=[tar_pos[1]],
            z=[tar_pos[2]],
            mode='markers',
            marker=dict(
                size=4,
                color='red',
                opacity=0.5
            )
        ),
        go.Scatter3d(
            x=[temp_pos[0]],
            y=[temp_pos[1]],
            z=[temp_pos[2]],
            mode='markers',
            marker=dict(
                size=4,
                color='black',
                opacity=0.5
            )
        )

    ])

    fig.show()


def get_closest_vertices(target, predicted):
    # Expand dimensions to calculate pairwise distances
    predicted = predicted.unsqueeze(2)  # Shape: (batch, num_verts, 1, 3)
    target = target.unsqueeze(1)  # Shape: (batch, 1, num_verts, 3)

    # Compute squared Euclidean distances
    distances_squared = torch.sum((predicted - target) ** 2,
                                  dim=-1)  # Shape: (batch, num_verts, num_verts)

    # Find the indices of the minimum distances along the target_vertices dimension
    closest_indices = torch.argmin(distances_squared, dim=2)

    return closest_indices

def cosine_similarity_loss(pred_normals, gt_normals):
    # normalize vectors
    pred_normals = F.normalize(pred_normals, p=2, dim=-1)  # Normalize along the last dimension
    gt_normals = F.normalize(gt_normals, p=2, dim=-1)

    # Compute cosine similarity between corresponding normals
    cosine_loss = 1 - (pred_normals * gt_normals).sum(dim=-1).mean()

    return cosine_loss


if __name__ == "__main__":
    path1 = "../data/Couch_T1/triangle_meshes/mesh-f00022.obj"
    path2 = "../data/Couch_T1/triangle_meshes/mesh-f00025.obj"
    path3 = "../data/Couch_T1/template_mesh/mesh-f00001.obj"
    target_meshes = load_objs_as_meshes([path1, path2])
    predicted = load_objs_as_meshes([path3, path3])

    # test1
    predicted_sampled, normals_predicted = sample_points_from_meshes(predicted, 2500,
                                                                     return_normals=True)
    target_sampled, normals_target = sample_points_from_meshes(target_meshes, 2500,
                                                               return_normals=True)

    indices = get_closest_vertices(target_sampled, predicted_sampled)
    idx = 500
    print(indices[0, idx])
    pred_mesh_point = predicted_sampled[0][idx]
    tar_mesh_point = target_sampled[0][indices[0, idx]]
    mesh_plotly(target_meshes[0], predicted[0], ['Target', 'Template'], pred_mesh_point, tar_mesh_point)

    # test2
    indices = get_closest_vertices(target_sampled, target_sampled)
    batch_indices = torch.arange(2).unsqueeze(1).expand(-1, 2500)
    gt_normals = normals_target[batch_indices, indices]
    normals_loss = cosine_similarity_loss(normals_target, gt_normals)
    print(normals_loss)