import sys
import os

import cv2
from PIL import Image
import torch
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))
from datasets.dataset_meshes import DatasetMeshWithImages, collate_fn
from training.render import render_meshes
import torch
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader


def transform_meshes(meshes, centers, scales, device):
    transformed_verts = []
    for i, mesh in enumerate(meshes):
        verts = mesh.verts_packed()
        verts = verts * scales[i].to(device)
        verts = verts + centers[i].to(device)
        verts = verts/1000
        transformed_verts.append(verts.float())

    transformed_meshes = Meshes(verts=transformed_verts, faces=meshes.faces_list(), textures=meshes.textures)

    return transformed_meshes

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = '../data/Couch_T1'
    dataset = DatasetMeshWithImages(path, device=device)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                                  collate_fn=lambda b, device=device: collate_fn(b, device), drop_last=True)
    for batch in train_dataloader:
        target_meshes, template_vertices, template_faces, template_textures, images, camera_parameters, centers, scales, frames, num_points, num_faces = batch
        verts = [template_vertices[s][:num_points[s]] for s in range(2)]
        meshes = Meshes(verts=verts, faces=template_faces, textures=template_textures)

        transformed = transform_meshes(meshes, centers, scales, device)

        rendered_images = render_meshes(transformed, camera_parameters, device)

        for img in rendered_images:
            img = to_pil_image(img.cpu())
            #img.save('rendered_image.png')

            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.show()
        break
