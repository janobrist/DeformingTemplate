from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes

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


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # get deformed meshes
        frame = self.frames[idx]
        mesh_trg_Path = os.path.join(self.trg_root, f"mesh-f{frame}.obj")
        target_mesh = load_objs_as_meshes([mesh_trg_Path], device=self.device)

        # get template mesh
        mesh_src_Path = os.path.join(self.src_root, "mesh-f00001.obj")
        template_mesh = load_objs_as_meshes([mesh_src_Path], device=self.device)

        # transform meshes
        target_mesh, _, _ = self.transform_meshes(target_mesh)
        template_mesh, scales, centers = self.transform_meshes(template_mesh)

        # load images
        images = self.load_images(frame)

        # get camera parameters
        camera_parameters = self.get_camera_parameters()

        return {'target_mesh': target_mesh,
                'template_mesh': template_mesh,
                "images": images,
                "camera_parameters": camera_parameters,
                'frame': self.frames[idx],
                'centers': centers,
                'scales': scales}

    def load_mesh(self, path):
        mesh = trimesh.load(path)
        return mesh.vertices, mesh.faces

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
    features_verts_src = torch.zeros((len(data), maxPoints, 3)).to(device)
    images = torch.zeros((len(data)*len(data[0]['images']), 3, 224, 224)).to(device)
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
    textures_template = [item['template_mesh'].textures for item in data]
    textures = textures_template[0].join_batch(textures_template[1:])
    template_faces = [item['template_mesh'].faces_packed() for item in data]

    # target meshes
    verts = [item['target_mesh'].verts_packed() for item in data]
    faces = [item['target_mesh'].faces_packed() for item in data]
    target_meshes = Meshes(verts=verts, faces=faces)

    frame = [item['frame'] for item in data]
    camera_parameters = [item['camera_parameters'] for item in data]
    centers = [item['centers'] for item in data]
    scales = [item['scales'] for item in data]

    return target_meshes, features_verts_src, template_faces, textures, images, camera_parameters, centers, scales, frame, num_vertices_src, num_faces_src


def collate_fn_nofor(data, device):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    #print('data[0]: ', data[0])
    #device='cuda:0'
    maxPoints = 9000#9000
    maxFaces = 17000#17000
    #_, labels, lengths = zip(*data)
    #max_len = max(lengths)
    #n_ftrs = data[0][0].size(1)
    features_verts_trg = torch.zeros((len(data), maxPoints, 3))
    features_faces_trg = torch.zeros((len(data), maxFaces, 3))
    features_verts_src = torch.zeros((len(data), maxPoints, 3))
    features_faces_src = torch.zeros((len(data), maxFaces, 3))

    orig_verts_trg = []
    orig_verts_src = []
    orig_faces_trg = []
    orig_faces_src = []
    #labels = torch.tensor(labels)
    #lengths = torch.tensor(lengths)
    #print('len: ', len(data))
    for i in range(len(data)):
        num_points = data[i]['num_points']
        num_faces = data[i]['num_faces']
        
        verts_trg = data[i]['vertices_trg'].to(device)
        #print('verts_trg shape: ', verts_trg.shape)
        faces_trg = data[i]['faces_trg'].to(device)
        
        #print('faces trg shape: ', faces_trg.shape)
        #print('vertices trg shape: ', verts_trg.shape)
        features_verts_trg[i] = torch.cat((verts_trg, torch.zeros((maxPoints - num_points, 3)).to(device)), dim=0)
        features_faces_trg[i] = torch.cat((faces_trg, torch.zeros((maxFaces - num_faces, 3)).to(device)), dim=0)

        
        verts_src = data[i]['vertices_src'].to(device)
        faces_src = data[i]['faces_src'].to(device)
        #print('faces src shape: ', faces_src.shape)
        #num_points = data[i]['num_points']
        #num_faces = data[i]['num_faces']
        features_verts_src[i] = torch.cat((verts_src, torch.zeros((maxPoints - num_points, 3)).to(device)), dim=0)
        features_faces_src[i] = torch.cat((faces_src, torch.zeros((maxFaces - num_faces, 3)).to(device)), dim=0)

        orig_verts_src.append(verts_src)
        orig_verts_trg.append(verts_trg)
        orig_faces_src.append(faces_src)
        orig_faces_trg.append(faces_trg)

        #print('features shape: ', features_verts_src[i].shape)
        #print('faces: ', data[i]['faces_src'].shape)
    
    #print('features shape: ', features_verts_src.shape)

    verts_trg = features_verts_trg
    faces_trg = features_faces_trg

    #print('faces src 0: ', faces_src[0])
    #print('faces trg 0: ', faces_trg[0])

    verts_src = features_verts_src
    faces_src = features_faces_src

    #faces = torch.cat([el['faces'].unsqueeze(0) for el in data], dim=0)
    name = [el['name'] for el in data]
    centers = torch.cat([el['center_obj'].unsqueeze(0) for el in data], dim=0)
    scale_obj = [el['scale_obj'] for el in data]
    centers_src = torch.cat([el['center_src'].unsqueeze(0) for el in data], dim=0)
    scale_src = [el['scale_src'] for el in data]
    num_points = [el['num_points'] for el in data]
    num_faces = [el['num_faces'] for el in data]
    
    
    return orig_verts_trg, orig_faces_trg, orig_verts_src, orig_faces_src, {'vertices_trg': verts_trg, 'faces_trg': faces_trg, 'vertices_src': verts_src, 'faces_src': faces_src, 'name': name, 'center_obj':centers, 'scale_obj':scale_obj, 'num_points':num_points, 'num_faces':num_faces, 'center_src':centers_src, 'scale_src':scale_src}
