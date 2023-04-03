import os
import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset
from scipy.stats import ortho_group
import trimesh

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

class PointClouds(Dataset):

    def __init__(self, dataset_path, labels, is_training=False):
        """
        Arguments:
            is_training: a boolean.
        """

        paths = []
        for path, subdirs, files in os.walk(dataset_path):
            for name in files:
                #print('name: ', name)
                #print('files: ', files)
                p = os.path.join(path, name)
                #assert p.endswith('.ply')
                paths.append(p)
        
        def get_label(p):
            return p.split('/')[-2]
        
        #paths = [p for p in paths if get_label(p) in labels]
        self.is_training = is_training
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        """
        Returns:
            x: a float tensor with shape [3, num_points].
        """
        
        p = self.paths[i]
        x = load_ply(p)

        meanOf = x.mean(0)
        
        x -= x.mean(0)
        d = np.sqrt((x ** 2).sum(1)).max()
        x /= d

        if self.is_training:
            x = augmentation(x)
        
        # color = np.ones_like(x)
        # cloud = trimesh.PointCloud(vertices=x, colors=color)
        # cloud.show(background=[0,0,0,0])
        
        x = torch.FloatTensor(x).permute(1, 0)
        #print('in between: ', x.shape)

        # color = np.ones_like(x.permute(1, 0))
        # cloud = trimesh.PointCloud(vertices=x.permute(1, 0), colors=color)
        # cloud.show(background=[0,0,0,0])
        #print('path: ', p)
        return x, p.split('/')[-1].split('.')[-2], meanOf, d


def load_ply(filename):
    """
    Arguments:
        filename: a string.
    Returns:
        a float numpy array with shape [num_points, 3].
    """
    #ply_data = PlyData.read(filename)
    mesh_obj=trimesh.load(filename)
    verts_obj = mesh_obj.vertices
    if(filename.endswith('ply')):
        points = verts_obj
    else:
        faces_obj = mesh_obj.faces
        faces_obj = torch.tensor(faces_obj).float()#.to('cuda')
        verts_obj = torch.tensor(verts_obj).float()#.to('cuda')
        #print('verts shape: ', verts_obj.shape)
        #print('faces shape: ', faces_obj.shape)
        #faces_idx_obj = faces_obj.to(device)
        #verts_obj = verts_obj.to(device)
        #print('verts obj shape: ', verts_obj.shape)
        #print('faces obj shape: ', faces_obj.shape)
        mesh = Meshes(verts=list(verts_obj.reshape(1, -1, 3)), faces=list(faces_obj.reshape(1, -1, 3)))
        points = sample_points_from_meshes(mesh, 3000).reshape(-1, 3)
    #print('the shape of points: ', points.shape)
    #print('points.shape: ', points.shape)
    #verts_obj = torch.tensor(verts_obj).float()
    #points = ply_data['vertex']
    #points = np.vstack([points['x'], points['y'], points['z']]).T
    #print('points: ', points.shape)

    # color = np.ones_like(points)
    # cloud = trimesh.PointCloud(vertices=points, colors=color)
    # cloud.show(background=[0,0,0,0])
    return points#.astype('float32')


from scipy.stats import ortho_group

def augmentation(x):
    """
    Arguments:
        x: a float numpy array with shape [b, n, 3].
    Returns:
        a float numpy array with shape [b, n, 3].
    """

    #jitter = np.random.normal(0.0, 1e-2, size=x.shape)
    #x += jitter.astype('float32')

    # batch size
    #b = x.shape[0]

    # random rotation matrix
    # m = ortho_group.rvs(3)  # shape [b, 3, 3]
    # m = np.expand_dims(m, 0)  # shape [b, 1, 3, 3]
    # m = m.astype('float32')

    # x = np.expand_dims(x, 1)
    # x = np.matmul(x, m)
    # x = np.squeeze(x, 1)

    return x
