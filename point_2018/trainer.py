import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

#from model import Autoencoder
from loss import ChamferDistance
import trimesh
import numpy as np
import open3d as o3d

import sys
sys.path.append("..")
#from .. import modelAya
from modelAya import Autoencoder

class Trainer:

    def __init__(self, num_steps, device, folder):

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.network = Autoencoder(k=1024, num_points=3000).to(device)
        self.network = self.network.apply(weights_init).to(device)
        #path_encoder='./models-cars-donuts-pointResnetEncoder/run00.pth'
        #check_auto = torch.load(path_encoder)
        #self.network.load_state_dict(check_auto)
        self.folder=folder
        self.loss = ChamferDistance()
        self.optimizer = optim.Adam(self.network.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=1e-7)

    def train_step(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            a float number.
        """
        #print('x.shape: ', x.shape)
        # with torch.no_grad():
        #     pt = x[5:6,...].cpu().reshape(2048, 3)
        #     print('pt.shape: ', pt.shape)
        #     color = np.ones_like(pt)
        #     cloud = trimesh.PointCloud(vertices=pt, colors=color)
        #     cloud.show(background=[0,0,0,0])

        #pcd = o3d.geometry.PointCloud()
        #y = x[2,...].permute(1, 0)
        #pcd.points = o3d.utility.Vector3dVector(np.float32(y.cpu().numpy()))#.float32)
        #o3d.io.write_point_cloud('train.ply', pcd)
        _, x_restored = self.network(x)
        #print('shape of x_restorated: ', x_restored.shape)
        loss = self.loss(x, x_restored)
        # with torch.no_grad():


        #print('x shape: ', x.shape)
        #print('x_restorated shape: ', x_restored.shape)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def evaluate(self, x, e, p, mean, scale):

        with torch.no_grad():
            #print('x shape: ', x.shape)
            _, x_restored = self.network(x)
            #print('x_restored shape: ', x_restored.shape)
            loss = self.loss(x, x_restored)
            
            pcd = o3d.geometry.PointCloud()
            #print('x shape: ', x.shape)
            b, _,_= x.shape
            # if(e % 100 == 0):
            #     for i in range(0, b): 
            #         x_ = (x[i,...].permute(1,0) * scale[i].to('cuda')+ mean[i].to('cuda'))
            #         pcd.points = o3d.utility.Vector3dVector(np.float32(x_.cpu().numpy()))#.float32)
            #         o3d.io.write_point_cloud(self.folder+'/plies/file_'+str(e)+'_'+p[i]+'.ply', pcd)

            #         pcd = o3d.geometry.PointCloud()
            #         #print('x shape: ', x.shape)
            #         x_restored_ = (x_restored[i,...].permute(1,0) * scale[i].to('cuda')+ mean[i].to('cuda'))
            #         pcd.points = o3d.utility.Vector3dVector(np.float32(x_restored_.cpu().numpy()))#.float32)
            #         o3d.io.write_point_cloud(self.folder+'/plies/file_restorated'+str(e)+'_'+p[i]+'.ply', pcd)

        


            # import os 
            # #os.environ['PYOPENGL_PLATFORM'] = 'egl'
            # pt = x.cpu().numpy().reshape(2048, 3)
            # print('pt.shape: ', pt.shape)
            # color = np.ones_like(pt)
            # cloud = trimesh.PointCloud(vertices=pt, colors=color)
            # cloud.show(background=[0,0,0,0])

            # pt = x_restored[0,...].cpu().numpy().reshape(2048, 3)
            # print('pt.shape: ', pt.shape)
            # color = np.ones_like(pt)
            # cloud = trimesh.PointCloud(vertices=pt, colors=color)
            # cloud.show(background=[0,0,0,0])

        return loss.item()
