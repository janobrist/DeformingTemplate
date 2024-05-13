import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))
from datasets.dataset_meshes import DatasetMeshWithImages, collate_fn
from torch.utils.data import DataLoader
import torch

def check_shapes(target_meshes):
    for i in range(len(target_meshes.verts_list())):
        num_vertices = target_meshes.verts_list()[i].shape[0]
        num_verts_from_tex = target_meshes.textures.verts_uvs_list()[i].shape[0]
        assert num_vertices == num_verts_from_tex, f"Number of vertices and texture vertices do not match: {num_vertices} != {num_verts_from_tex}"

if __name__ == "__main__":
    path = '../data/Couch_T1'
    device = torch.device("cuda:0")
    dataset = DatasetMeshWithImages(path, device)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True,
                                  collate_fn=lambda b, device=device: collate_fn(b, device), drop_last=True)
    print("Number of batches:",  len(train_dataloader))
    print("Number of samples:", len(dataset))
    for batch in train_dataloader:
        target_mesh, template_mesh, images, camera_parameters, centers, scales, frames, num_vertices, num_faces = batch
        check_shapes(target_mesh)
        print("Frame number:", frames)
        print("Number of vertices in template mesh:", num_vertices)
        print("Template mesh vertices shape:", template_mesh.verts_list()[0].shape)
        print("Template mesh faces shape:", template_mesh.faces_list()[0].shape)
        print("Target mesh vertices shape:", [mesh.shape for mesh in target_mesh.verts_list()])
        print("Target mesh faces shape:", target_mesh.faces_list()[0].shape)
        print("Image size:", images[0][0].shape)
        print("Camera parameters:", camera_parameters[0])
        # for i in range(len(sample['images'])):
        #     img = sample['images'][i]
        #     plt.imshow(img.numpy().transpose(1, 2, 0))  # Convert CxHxW to HxWxC
        #     plt.axis('off')  # Turn off axis numbers and ticks
        #     plt.show()
        break
