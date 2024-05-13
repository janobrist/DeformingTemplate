import torch
from PIL import Image
from torchvision import transforms
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)
import sys
from pympler import asizeof
from torchvision import transforms


def print_memory_usage(local_vars):
    for name, value in local_vars.items():
        print(
            f'{name} uses {asizeof.asizeof(value)} bytes (deep size) and {sys.getsizeof(value)} bytes (shallow size)')


def process_rendered_image(image):
    # Increase saturation and brightness of the rendered image
    image = image ** (1 / 2.2)
    rgb_tensor = image[:, :, :, :3]

    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
    adjusted_tensor = transforms.functional.adjust_saturation(rgb_tensor, 2)
    adjusted_tensor = transforms.functional.adjust_brightness(adjusted_tensor, 1.05)

    return adjusted_tensor

def transform_mesh(mesh, scales, centers, device):
    offsets = torch.zeros((mesh.verts_packed().shape[0], 3)).to(device)
    total_verts = 0
    boundaries = mesh.get_bounding_boxes()
    for i, verts in enumerate(mesh.verts_list()):
        center = centers[i].to(device)
        scale = scales[i]
        offsets[total_verts:total_verts + verts.shape[0], :] -= center
        total_verts += verts.shape[0]

    mesh.offset_verts_(offsets)
    mesh.scale_verts_(scales)

    return mesh

# def render_meshes(mesh, camera_parameters, device):
#     tex = mesh.textures.clone()
#     print_memory_usage(locals())
#     mesh = mesh.extend(3).to(device)
#     rotations, translations = [], []
#     for extrinsic_matrix in camera_parameters['extrinsics']:
#         rotation = extrinsic_matrix[:3, :3]
#         rotations.append(rotation)
#         translations.append(extrinsic_matrix[:3, 3])
#
#     camera_matrices = []
#     sizes = torch.zeros((4, 2)).to(device)
#     for i, intrinsic_matrix in enumerate(camera_parameters['intrinsics']):
#         intrinsic_matrix[0, 2] -= 1440-512
#         intrinsic_matrix[1, 2] -= 2600-512
#         camera_matrices.append(intrinsic_matrix)
#         sizes[i, 0] = 1024
#         sizes[i, 1] = 1024
#
#     rotations = torch.stack(rotations).to(device).float()
#     translations = torch.stack(translations).to(device).float()
#     camera_matrices = torch.stack(camera_matrices).to(device).float()
#     cameras = cameras_from_opencv_projection(rotations, translations, camera_matrices, sizes).to(device)
#
#     raster_settings = RasterizationSettings(
#         image_size=1024,
#         blur_radius=0.0
#     )
#
#     renderer = MeshRenderer(
#         rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
#         shader=HardPhongShader(device=device, cameras=cameras)
#     )
#     rendered_images = renderer.forward(meshes)
#
#     output_images = []
#     for image in rendered_images:
#         output_images.append(process_rendered_image(image))
#
#
#     return torch.stack(output_images)


def render_meshes(meshes, camera_parameters, device):
    transformation = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256
        transforms.CenterCrop(224),  # Crop a 224x224 patch from the center
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalize with ImageNet's mean and std
    ])
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0
    )
    rendered_images = []
    for i, mesh in enumerate(meshes):
        for extrinsic_matrix, intrinsic_matrix in zip(camera_parameters[i]['extrinsics'], camera_parameters[i]['intrinsics']):
            rotation = torch.tensor(extrinsic_matrix[:3, :3]).unsqueeze(0).float()
            translation = torch.tensor(extrinsic_matrix[:3, 3]).unsqueeze(0).float()

            intrinsic_matrix[0, 2] -= 1440-512
            intrinsic_matrix[1, 2] -= 2600-512
            camera_matrix = torch.tensor(intrinsic_matrix).unsqueeze(0).float()
            image_size = torch.tensor([1024, 1024]).unsqueeze(0)

            camera = cameras_from_opencv_projection(rotation, translation, camera_matrix, image_size).to(device)

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
                shader=HardPhongShader(device=device, cameras=camera)
            )
            rendered_image = renderer.forward(mesh)
            rendered_image = process_rendered_image(rendered_image)
            rendered_images.append(transformation(rendered_image))

    rendered_images = torch.stack(rendered_images).squeeze(1)

    return rendered_images

