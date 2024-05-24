import pyvista as pv
import pymeshfix as mf
import open3d as o3d
import os

def reapir_mesh(mesh_path, output_path):
    # read mesh wit pyvista
    mesh = pv.read(mesh_path)
    # repair mesh
    meshfix = mf.MeshFix(mesh)
    meshfix.repair(verbose=False, joincomp=True, remove_smallest_components=False)

    # convert mesh to open3d
    vertices = meshfix.mesh.points
    faces = meshfix.mesh.faces.reshape(-1, 4)[:, 1:4]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # remove non-manifold edges
    mesh = mesh.remove_non_manifold_edges()

    print(mesh.is_watertight())
    o3d.io.write_triangle_mesh(output_path, mesh)

if __name__ == "__main__":
    shot = "Paper"
    path = f"../data/{shot}"
    for directory in os.listdir(path):
        mesh_path = os.path.join(path, directory, "template_mesh")
        for file in os.listdir(mesh_path):
            if file.endswith(".obj"):
                mesh_path = os.path.join(mesh_path, file)
                break
        output_path = os.path.join(path, directory, "template_mesh", "repaired.obj")
        reapir_mesh(mesh_path, output_path)