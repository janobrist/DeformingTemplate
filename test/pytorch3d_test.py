from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from torchvision import transforms
from pytorch3d.io import save_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    AmbientLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)




if __name__ == "__main__":
    mesh_path1 = '../data/Couch_T1/mesh-f00001.obj'
    mesh_path2 = '../data/Couch_T1/mesh-f00002.obj'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mesh = load_objs_as_meshes([mesh_path1, mesh_path2], device=device)

    offsets = torch.zeros((mesh.verts_packed().shape[0], 3)).to(device)
    scales = torch.zeros((2, 1)).to(device)
    total_verts = 0
    boundaries = mesh.get_bounding_boxes()
    for i, verts in enumerate(mesh.verts_list()):
        center = torch.tensor([0.5 * (boundaries[i][0][0] + boundaries[i][0][1]),
                               0.5 * (boundaries[i][1][0] + boundaries[i][1][1]),
                               0.5 * (boundaries[i][2][0] + boundaries[i][2][1])]).to(device)
        scale = torch.sqrt((verts ** 2).sum(1)).max()
        offsets[total_verts:total_verts + verts.shape[0], :] -= center
        scales[i] = 1/scale
        total_verts += verts.shape[0]

    mesh.offset_verts_(offsets)
    mesh.scale_verts_(scales)

    out_path = ["test1.obj", "test2.obj"]
    counter = 0
    for faces, verts in zip(mesh.faces_list(), mesh.verts_list()):
        save_obj(out_path[counter], verts, faces)
        counter += 1

    print("Done")




