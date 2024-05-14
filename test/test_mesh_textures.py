from pytorch3d.renderer import MeshRenderer
import matplotlib.pyplot as plt
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
import trimesh
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)
from pytorch3d.io import load_objs_as_meshes

if __name__ == "__main__":
    device = "cpu"
    mesh_path = "../data/Couch_T1/triangle_meshes/mesh-f00012.obj"
    mesh = trimesh.load(mesh_path)
    colors = torch.tensor(mesh.visual.to_color().vertex_colors).unsqueeze(0)
    colors = colors[:, :, :3]
    textures = TexturesVertex(colors)
    verts = torch.tensor(mesh.vertices).float()
    faces = torch.tensor(mesh.faces).float()

    mesh_torch = Meshes(verts=[verts], faces=[faces], textures=textures)
    mesh_torch2 = load_objs_as_meshes([mesh_path])


    # Initialize an OpenGL perspective camera.
    R, T = look_at_view_transform(4, 1, 50)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters.
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a Phong renderer by composing a rasterizer and a shader. Here we can use a predefined
    # PhongShader, passing in the device on which to initialize the default parameters
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras)
    )

    image = renderer.forward(mesh_torch2)

    image = image[0]
    plt.imshow(image.numpy())  # Convert CxHxW to HxWxC
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
