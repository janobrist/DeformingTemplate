from pytorch3d.io import load_objs_as_meshes
import os
import torch


if __name__ == "__main__":
    path = "../data"
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir, 'template_mesh')
        for file in os.listdir(dir_path):
            if file.endswith(".obj"):
                file_path = os.path.join(dir_path, file)
                mesh = load_objs_as_meshes([file_path])
                try:
                    torch.save({'maps': mesh.textures.maps_padded(), 'verts_uvs': mesh.textures.verts_uvs_padded(),
                            'faces_uvs': mesh.textures.faces_uvs_padded()}, f'{dir_path}/textures.pth')
                except Exception as e:
                    continue


