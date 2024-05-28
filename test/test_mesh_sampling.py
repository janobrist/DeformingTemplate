from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_objs_as_meshes
import torch
import trimesh
from pytorch3d.io import load_objs_as_meshes
from scipy.spatial import KDTree


def compute_avg_distance(sampled_points):
    # Convert sampled points to numpy array for KDTree
    print(sampled_points.shape)
    sampled_points_np = sampled_points[0].cpu().numpy()

    # Build KDTree
    kdtree = KDTree(sampled_points_np)

    # Query for the nearest neighbor of each point (excluding the point itself)
    distances, _ = kdtree.query(sampled_points_np, k=2)

    # The nearest neighbor distance is the second column in the distances array
    nearest_distances = distances[:, 1]

    # Compute the average distance
    avg_distance = nearest_distances.mean()

    print(f"Average distance to the nearest neighbor: {avg_distance}")
    return avg_distance

def load_mesh(path):
    # load
    mesh = trimesh.load(path)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.float32)

    # transform
    center = verts.mean(0)
    verts = verts - center
    scale = torch.sqrt((verts ** 2).sum(1)).max()
    verts = verts / scale

    return verts, faces, center, scale



if __name__ == "__main__":
    mesh_path = "../data/Paper/Paper_T1_Poke1/template_mesh/mesh-f00001.obj"
    verts, faces, center, scale = load_mesh(mesh_path)
    mesh = Meshes(verts=[verts], faces=[faces])

    sampled = sample_points_from_meshes(mesh, 5000)
    compute_avg_distance(sampled)