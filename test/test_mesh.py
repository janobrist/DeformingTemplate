import open3d as o3d
import numpy as np
import trimesh
import meshlib.mrmeshpy as mrmeshpy
import pyvista as pv
import pymeshfix as mf
import imageio.v2 as imageio

def fill_holes(mesh_path):
    # Load mesh
    mesh = mrmeshpy.loadMesh(mesh_path)

    # Find single edge for each hole in mesh
    hole_edges = mesh.topology.findHoleRepresentiveEdges()
    # edge_path = mrmeshpy.vectorEdgePath()
    # edge_path.append(hole_edges)
    # hole_vert = mrmeshpy.findHoleVertIdsByHoleEdges(mesh.topology, edge_path)
    # hole_vert = hole_vert.pop()
    # indexes = []
    # for id in hole_vert:
    #     index = id.get()
    #     indexes.append(index)
    # hole_verts = np.array(indexes)

    for e in hole_edges:
        #  Setup filling parameters
        params = mrmeshpy.FillHoleParams()
        params.metric = mrmeshpy.getUniversalMetric(mesh)
        #  Fill hole represented by `e`
        mrmeshpy.fillHole(mesh, e, params)

    mrmeshpy.saveMesh(mesh, 'test2.obj')
def convert_obj_format(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    with open(output_path, 'w') as file:
        for line in lines:
            if line.startswith('f '):
                # Split the face line into components
                parts = line.split()
                # Process each vertex index to the new format
                new_parts = [
                    f"{index}/{index}/{index}" for index in parts[1:]
                ]
                # Write the new face line
                new_line = "f " + " ".join(new_parts) + "\n"
                file.write(new_line)
            else:
                # Write the line as is if it's not a face line
                file.write(line)



if __name__ == "__main__":
    # read mesh wit pyvista
    mesh_path = "../data/Couch_T3/template_mesh/mesh-f00001.obj"
    #mesh_path = "simplified.ply"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("Is mesh watertight:", mesh.is_watertight())
    mesh = pv.read(mesh_path)


    # repair mesh
    meshfix = mf.MeshFix(mesh)
    holes = meshfix.extract_holes()
    p = pv.Plotter()
    p.add_mesh(mesh, color=True)
    p.add_mesh(holes, color="r", line_width=8)
    p.enable_eye_dome_lighting()  # helps depth perception
    p.show()
    meshfix.repair(verbose=True, joincomp=True, remove_smallest_components=False)

    # convert mesh to open3d
    vertices = meshfix.mesh.points
    faces = meshfix.mesh.faces.reshape(-1, 4)[:, 1:4]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # remove non-manifold edges
    mesh = mesh.remove_non_manifold_edges()

    print(mesh.is_watertight())
    o3d.io.write_triangle_mesh("../data/Couch_T1/template_mesh/test.obj", mesh)







