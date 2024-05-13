import sys
import os

import cv2
from PIL import Image
import torch
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../'))
from datasets.dataset_meshes import DatasetMeshWithImages
from training.render import render_meshes
import torch
from torchvision.transforms.functional import to_pil_image


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = '../data/Couch_T2'
    dataset = DatasetMeshWithImages(path, device=device)
    sample = dataset[0]
    camera_parameters = sample['camera_parameters']

    target_mesh = sample['target_mesh']
    print(target_mesh.textures)
    rendered_images = render_meshes(target_mesh, camera_parameters, device)


    for img in rendered_images:
        new_img = img.permute(2, 0, 1)
        print(new_img.shape)
        img = to_pil_image(new_img.cpu())
        img.save('rendered_image.png')

        plt.imshow(img)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()
