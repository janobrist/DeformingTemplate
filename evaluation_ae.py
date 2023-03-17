import torch
import numpy as np
from foldingNet_model import AutoEncoder
#from chamfer_distance.chamfer_distance import ChamferDistance
from dataset import PointClouds
import trimesh 
import os
import open3d as o3d
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)


test_dataset = PointClouds('/home/elham/Desktop/makeDataset/warping/warping_shapes_generation/build_path/ycb_mult_5_one_seq/val', is_training=True)

#test_dataset = ShapeNetPartDataset(root='/home/rico/Workspace/Dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
#                                   npoints=2048, split='test', classification=False, data_augmentation=True)

#Dataset_mesh_objects()

#test_dataset = Dataset_mesh_objects(trg_root='./car-donut-train', src_root='./srcMeshes')
#train_dataloader = DataLoader(training_dataset, batch_size=B, shuffle=True, collate_fn=collate_fn)

#test_dataset = 

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
modelFoldingNet = AutoEncoder()
folder='/home/elham/Desktop/FoldingNet/first_50_each'
os.makedirs(folder+'/plies/', exist_ok=True)
dict = torch.load(folder+'/logs/model_epoch_9000.pth')
modelFoldingNet.load_state_dict(dict["model_state_dict"])
device = torch.device('cuda')
modelFoldingNet.to(device)

#cd_loss = ChamferDistance()

# evaluation
modelFoldingNet.eval()
total_cd_loss = 0

with torch.no_grad():
    print('length of the test dataset: ', len(test_dataset))
    allLosses=[]
    id = 0
    for data, p, mean, scale in test_dataloader:
        #print('data.shape: ', data.shape)
        #print('p: ', p)
        #if(id > 1):
        #    break
        point_clouds = data
        b, _, _ = point_clouds.shape
        point_clouds = point_clouds#.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        print('pointclouds shape: ', point_clouds.shape)
        recons = modelFoldingNet(point_clouds)
        #print('recons shape: ', recons[0,...].permute(1,0).shape)
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        #color = np.ones_like(recons[0,...].permute(1,0).cpu().numpy())
        #cloud = trimesh.PointCloud(vertices=recons[0,...].permute(1,0).cpu().numpy(), colors=color)
        #cloud.show(background=[0,0,0,0])
        pcd = o3d.geometry.PointCloud()
        #print('x shape: ', x.shape)
        for i in range(b):
            x_restored_ = (recons[i,...].permute(1,0) * scale[i].to('cuda')+ mean[i].to('cuda'))
            pcd.points = o3d.utility.Vector3dVector(np.float32(x_restored_.cpu().numpy()))#.float32)
            o3d.io.write_point_cloud(folder+'/plies/fold_'+p[i]+'.ply', pcd)
            ls = chamfer_distance(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            allLosses.append(ls[0].cpu())
            print(ls[0].cpu())
            total_cd_loss += ls[0].cpu()
        id+=1
    np.savetxt('errors_04379243.txt', allLosses, delimiter=',') 

# calculate the mean cd loss
mean_cd_loss = total_cd_loss / len(test_dataset)
print('Mean Chamfer Distance of all Point Clouds:', mean_cd_loss)
