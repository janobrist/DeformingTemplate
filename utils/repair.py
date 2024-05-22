import pyvista as pv
import pymeshfix as mf
import open3d as o3d

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
    num_takes = 10
    shot = "Couch"
    for i in range(num_takes):
        mesh_path = f"../data/{shot}_T{i+1}/template_mesh/mesh-f00001.obj"
        output_path = f"../data/{shot}_T{i+1}/template_mesh/mesh-f00001_repaired.obj"
        reapir_mesh(mesh_path, output_path)