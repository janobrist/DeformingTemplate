from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV
import time
import json

class Dataset_mesh(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.samples = []
        i = 0

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            print("WARNING: CPU only, this will be slow!")

        for meshFile in os.listdir(data_root):
            meshPath = os.path.join(data_root, meshFile)
            mesh_obj = trimesh.load(meshPath)
            verts_obj = mesh_obj.vertices
            faces_obj = mesh_obj.faces

            faces_obj = torch.tensor(faces_obj).float()
            verts_obj = torch.tensor(verts_obj).float()
            #print('verts shape: ', verts_obj.shape)
            #print('faces shape: ', faces_obj.shape)
            faces_idx_obj = faces_obj.to(device)
            verts_obj = verts_obj.to(device)
            center_obj = verts_obj.mean(0)

            verts_obj = verts_obj - center_obj

            #x -= x.mean(0)
            #d = np.sqrt((x ** 2).sum(1))
            #x /= d.max()

            #scale_obj = max(verts_obj.abs().max(0)[0])
            #verts_obj = verts_obj/scale_obj

            scale_obj = torch.sqrt((verts_obj ** 2).sum(1)).max()
            #print('###############scale_obj: ', scale_obj)
            verts_obj = verts_obj/scale_obj


            self.samples.append({'vertices': verts_obj, 'faces': faces_idx_obj, 'name': meshFile, 'center_obj': center_obj, 'scale_obj':scale_obj})

        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DatasetMeshWithImages(Dataset):
    def __init__(self, root_dir, device):
        self.trg_root = f"{root_dir}/triangle_meshes"
        self.src_root = f"{root_dir}/template_mesh"
        self.image_root = f"{root_dir}/images"
        self.calib_root = f"{root_dir}/calibration"
        self.name = os.path.basename(root_dir)
        self.camera_names = [name for name in sorted(os.listdir(self.image_root))]
        self.frames = []
        self.device = device
        self.transformation = transforms.Compose([
            transforms.Resize(256),  # Resize the image to 256x256
            transforms.CenterCrop(224),  # Crop a 224x224 patch from the center
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Normalize with ImageNet's mean and std
        ])
        for mesh_file in sorted(os.listdir(self.trg_root)):
            if mesh_file.endswith('.obj'):
                self.frames.append(mesh_file[-9:-4])

        self.robot_data = []
        # robot_data
        with open(os.path.join(root_dir, "robot_data.json"), 'r') as f:
            data = json.load(f)

        for frame in self.frames:
            for item in data['data']:
                if int(item['frame']) == int(frame):
                    self.robot_data.append(item)


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        # get deformed meshes
        target_mesh_path = os.path.join(self.trg_root, f"mesh-f{frame}.obj")
        target_mesh_vertices, target_mesh_faces, center_target, scale_target = self.load_mesh(target_mesh_path)
        target_mesh = Meshes(verts=[target_mesh_vertices], faces=[target_mesh_faces]).to(self.device)


        # get template mesh
        template_mesh_path = os.path.join(self.src_root, "mesh-f00001_repaired.obj")
        template_mesh_vertices, template_mesh_faces, center_template, scale_template = self.load_mesh(template_mesh_path)
        #template_tex_tensor = torch.load(os.path.join(self.src_root, "textures.pth"))
        #template_mesh_textures = TexturesUV(maps=template_tex_tensor['maps'], verts_uvs=template_tex_tensor['verts_uvs'], faces_uvs=template_tex_tensor['faces_uvs'])

        #template_mesh = Meshes(verts=[template_mesh_vertices], faces=[template_mesh_faces], textures=template_mesh_textures).to(self.device)
        template_mesh = Meshes(verts=[template_mesh_vertices], faces=[template_mesh_faces]).to(self.device)

        # load images
        images = self.load_images(frame)


        # transform ee pos in robot data
        robot_data = self.robot_data[idx]
        ee_pos = torch.tensor(1000*np.array(robot_data['T_ME'])[:3, 3])
        ee_pos = (ee_pos - center_target)/scale_target
        robot_data['ee_pos'] = ee_pos

        # get camera parameters
        camera_parameters = self.get_camera_parameters()


        return {'target_mesh': target_mesh,
                'template_mesh': template_mesh,
                "images": images,
                "camera_parameters": camera_parameters,
                "robot_data": robot_data,
                'frame': self.frames[idx],
                'centers': center_template,
                'scales': scale_template,
                "name": self.name}

    def load_mesh(self, path):
        # load
        mesh = trimesh.load(path)
        print(mesh.is_watertight)
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.float32)

        # transform
        center = verts.mean(0)
        verts = verts - center
        scale = torch.sqrt((verts ** 2).sum(1)).max()
        verts = verts / scale

        return verts, faces, center, scale

    def load_images(self, frame):
        images = []
        for camera in self.camera_names:
            image_path = os.path.join(self.image_root, camera, f"{frame}.png")
            image = Image.open(image_path)
            image = self.transformation(image)
            images.append(image)

        return images

    def get_camera_parameters(self):
        rgb_cameras = dict(np.load(f'{self.calib_root}/cameras/rgb_cameras.npz'))
        extrinsics = []
        intrinsics = []
        for camera in self.camera_names:
            for k, id in enumerate(rgb_cameras['ids']):
                if id == int(camera):
                    idx = k
                    break
            extrinsics.append(torch.tensor(rgb_cameras['extrinsics_ori'][idx]))
            intrinsics.append(torch.tensor(rgb_cameras['intrinsics_ori'][idx]))

        return {'extrinsics': extrinsics, 'intrinsics': intrinsics}

    def transform_meshes(self, mesh):
        offsets = torch.zeros((mesh.verts_packed().shape[0], 3)).to(self.device)
        scales = torch.zeros((len(mesh.verts_list()), 1)).to(self.device)
        centers = []
        total_verts = 0
        boundaries = mesh.get_bounding_boxes()
        for i, verts in enumerate(mesh.verts_list()):
            center = torch.tensor([0.5 * (boundaries[i][0][0] + boundaries[i][0][1]),
                                   0.5 * (boundaries[i][1][0] + boundaries[i][1][1]),
                                   0.5 * (boundaries[i][2][0] + boundaries[i][2][1])]).to(self.device)
            scale = torch.sqrt((verts ** 2).sum(1)).max()
            offsets[total_verts:total_verts + verts.shape[0], :] -= center
            scales[i] = 1 / scale
            total_verts += verts.shape[0]
            centers.append(center)

        mesh.offset_verts_(offsets)
        mesh.scale_verts_(scales)

        return mesh, scales, centers

def collate_fn(data, device):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    # initialize tensors
    maxPoints = 9000#16000
    features_verts_src = torch.zeros((len(data), maxPoints, 3), dtype=torch.float32).to(device)
    images = torch.zeros((len(data)*len(data[0]['images']), 3, 224, 224), dtype=torch.float32).to(device)
    num_vertices_src = []
    num_faces_src = []
    # adjust size of tensors
    for i in range(len(data)):
        template_mesh = data[i]['template_mesh']
        verts_src = template_mesh.verts_packed()
        faces_src = template_mesh.faces_packed()
        num_vertices_src.append(verts_src.shape[0])
        num_faces_src.append(faces_src.shape[0])

        features_verts_src[i] = torch.cat((verts_src, torch.zeros((maxPoints - verts_src.shape[0], 3)).to(device)), dim=0)

        # images
        for j in range(len(data[i]['images'])):
            images[i*len(data[i]['images']) + j] = data[i]['images'][j].to(device)


    # template meshes
    # textures_template = [item['template_mesh'].textures for item in data]
    # textures = textures_template[0].join_batch(textures_template[1:])
    template_faces = [item['template_mesh'].faces_packed() for item in data]

    # target meshes
    verts = [item['target_mesh'].verts_packed() for item in data]
    faces = [item['target_mesh'].faces_packed() for item in data]
    target_meshes = Meshes(verts=verts, faces=faces)

    frame = [item['frame'] for item in data]
    camera_parameters = [item['camera_parameters'] for item in data]
    centers = [item['centers'] for item in data]
    scales = [item['scales'] for item in data]
    names = [item['name'] for item in data]
    robot_data = [item['robot_data'] for item in data]

    return target_meshes, features_verts_src, template_faces, images, camera_parameters, centers, scales, frame, num_vertices_src, names, robot_data



