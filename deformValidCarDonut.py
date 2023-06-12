import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from nvp_cadex import NVP_v2_5_frame
from torch.utils.data import Dataset
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import trimesh
from dataset_meshes_test import Dataset_mesh, Dataset_mesh_objects, collate_fn
from torch.utils.data import DataLoader
import random
from foldingNet_model import AutoEncoder
import torch.nn.init as init
from modelAya import Autoencoder
import open3d as o3d
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


config='1'
moreFaces=False
B = 1


parser = argparse.ArgumentParser(description="Just an example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--load", action="store_true", help="checksum blocksize")
parser.add_argument('--k', "--k", type=int, default=1024)
parser.add_argument('--encoder_type', type=str, default='folding')

args = vars(parser.parse_args())

#0.0006
if(config == "0"):
    deformed_model = '/home/elham/srl-nas/elham/research_project/logs/nvp_2018_1024dim_carDonut_cosinusAneal_End2End/'
    #path_autoencoder='/home/elham/Desktop/point-cloud-autoencoder/auto2018_1024dim_3000points_NoAug_1seq_5ycb/models/check_min.pt'
    trg_root='/home/elham/hdd/data/car_donut_data/train_car/'
    src_root='/home/elham/hdd/data/car_donut_data/test_in/'
    args["k"]=1024
    coeff = 8
#0.0006
elif(config == "1"):
    deformed_model = '/home/elham/srl-nas/elham/research_project/logs/nvp_2018_1024dim_carDonut_cosinusAneal_End2End_ONLY/'
    #path_autoencoder='/home/elham/Desktop/point-cloud-autoencoder/auto2018_1024dim_3000points_NoAug_1seq_5ycb/models/check_min.pt'
    trg_root='/home/elham/hdd/data/car_donut_data/train_car/'
    src_root='/home/elham/hdd/data/car_donut_data/test_in/'
    args["k"]=1024
    coeff = 8

    

path_load_check_decoder = deformed_model+'check/'+ 'check_min'+'.pt'
os.makedirs(deformed_model+ 'check', exist_ok=True)
os.makedirs(deformed_model + 'meshes_valid_morefaces', exist_ok=True)
os.makedirs(deformed_model + 'meshes_valid', exist_ok=True)
os.makedirs(deformed_model + 'meshes_trg_val', exist_ok=True)
os.makedirs(deformed_model + 'meshes_src_val', exist_ok=True)
os.makedirs(deformed_model + 'gt_sampled_pcl', exist_ok=True)
os.makedirs(deformed_model + 'meshes_compare_deform_decode', exist_ok=True)


device='cuda:0'
valid_dataset = Dataset_mesh_objects(trg_root=trg_root, src_root=src_root, moreFaces=moreFaces)
if(config=="8"):
    valid_dataloader = DataLoader(valid_dataset, batch_size=B, shuffle=True, collate_fn=lambda b, device=device: collate_fn(b, device, moreFaces), drop_last=True)
else:
    valid_dataloader = DataLoader(valid_dataset, batch_size=B, shuffle=False, collate_fn=collate_fn)

if(config == "0" or config == "1" or config == "5" or config == "6" or config == "7" or config == "8"):
    args['encoder_type'] = "2018"
else:
    args['encoder_type'] = "folding"
print('args: ', args['load'])

print('k: ', args['k'])
print('encoder: ', args['encoder_type'])
# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

print('device: ', device)

writer = SummaryWriter(log_dir=deformed_model+'events/')
###########################################################################

#homeomorphism_decoder:
n_layers = 6
#dimension of the code
feature_dims = 128 #2^7 * 2^3
hidden_size = [128, 64, 32, 32, 32]
#the dimension of the coordinates to be projected onto
proj_dims = 128
code_proj_hidden_size = [128, 128, 128]
proj_type = 'simple'
block_normalize = True
normalization = False

#print('here1')
c_dim = 128
hidden_dim = 128

homeomorphism_decoder = NVP_v2_5_frame(n_layers=n_layers, feature_dims=feature_dims*coeff, hidden_size=hidden_size, proj_dims=proj_dims,\
code_proj_hidden_size=code_proj_hidden_size, proj_type=proj_type, block_normalize=block_normalize, normalization=normalization).to('cuda')



numOfPoints=3000
if(args['encoder_type'] == '2018'):
    network = Autoencoder(k=args['k'], num_points=numOfPoints).to(device)
elif(args['encoder_type'] == 'folding'):
    network = AutoEncoder(k=args['k'])
device = torch.device('cuda')
network.to(device)


if(not 'End' in deformed_model):
    check_auto = torch.load(path_autoencoder, map_location='cuda:0')
    #print('check_auto: ', check_auto['model_state_dict'].keys())
    if(args['encoder_type'] == '2018'):
        network.load_state_dict(check_auto["model"])
    else:
        #print(check_auto["model_state_dict"])
        network.load_state_dict(check_auto["model_state_dict"])



if(args['load']):
    checkpoint = torch.load(path_load_check_decoder, map_location='cuda:0')
    #homeomorphism_encoder.load_state_dict(checkpoint['encoder'])
    homeomorphism_decoder.load_state_dict(checkpoint['decoder'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    if('End' in deformed_model):
        print('in end')
        if(args['encoder_type'] == '2018'):
            network.load_state_dict(checkpoint['network'])
        else:
            #print(check_auto["model_state_dict"])
            network.load_state_dict(checkpoint["network"])


Nepochs = 500000

w_chamfer = 1.0
#print('here3')
w_edge = 0 #1.0

w_normal = 1.0 #0.01

w_laplacian = 0 #1.0 #0.1

plot_period = 250

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


iteration = 0
done = False

losses = []
lossIndividus = {}
keys = {'scissor', 'bleach', 'hammer', 'orange', 'brick', 'dice'}

radii = {'scissors':None, 'bleach':None, 'hammer':None, 'orange':None, 'foam':None, 'dice':None}
radiiEx = {'scissors':True, 'bleach':True, 'hammer':True, 'orange':True, 'foam':True, 'dice':True}
for key in keys:
    lossIndividus[key]=[]
#print('epoch:', epoch)
homeomorphism_decoder.eval()
network.eval()
import time
for i, item in enumerate(valid_dataloader):
    begin= time.time()
    if(config=="8"):
        orig_verts_trg, orig_faces_trg, orig_verts_src, orig_faces_src, changedItem=item
        num_points = changedItem['num_points']
        #num_faces = changedItem['num_faces']
        B,_,_ = changedItem['vertices_src'].shape

        trg_mesh_verts_rightSize = orig_verts_trg #[item['vertices_trg'][s][:num_points[s]]for s in range(B)]

        trg_mesh_faces_rightSize = orig_faces_trg #[item['faces_trg'][s][:num_faces[s]]for s in range(B)]


        src_mesh_verts_rightSize = orig_verts_src #[item['vertices_src'][s][:num_points[s]]for s in range(B)]

        src_mesh_faces_rightSize = orig_faces_src #[item['faces_src'][s][:num_faces[s]]for s in range(B)]
    else:
        num_points = item['num_points']
        num_faces = item['num_faces']
        B,_,_ = item['vertices_src'].shape
        num_points=15018
        num_faces=30032
        trg_mesh_verts_rightSize = [item['vertices_trg'][s][:num_points]for s in range(B)]
        trg_mesh_faces_rightSize = [item['faces_trg'][s][:num_faces]for s in range(B)]

        if(moreFaces):
            print('in more')
            num_points=27588
            num_faces=55172
        src_mesh_verts_rightSize = [item['vertices_src'][s][:num_points]for s in range(B)]
        src_mesh_faces_rightSize = [item['faces_src'][s][:num_faces]for s in range(B)]



    trg_mesh = Meshes(verts=trg_mesh_verts_rightSize, faces=trg_mesh_faces_rightSize)
    src_mesh = Meshes(verts=src_mesh_verts_rightSize, faces=src_mesh_faces_rightSize)
    
    seq_pc_trg = sample_points_from_meshes(trg_mesh, numOfPoints).to('cuda')

    seq_pc_src = sample_points_from_meshes(src_mesh, numOfPoints).to('cuda')

    begCode= time.time()
    with torch.no_grad():
        #print('seq_pc_trg shape: ', seq_pc_trg.shape)
        if(args['encoder_type'] == '2018'):
            code_trg , _ = network(seq_pc_trg.permute(0, 2, 1))
        else:
            code_trg = network.encoder(seq_pc_trg.permute(0, 2, 1))
        #code_trg = network.encoder(seq_pc_trg.permute(0, 2, 1))
    endCode= time.time()
    print('time coding: ', endCode-begCode)
    
    #print('code_trg.shape: ', code_trg.shape)

    b, k = code_trg.shape
    if(config=="8"):
        query = changedItem['vertices_src'].to('cuda')
    else:
        query = item['vertices_src'].to('cuda')

    #print('code trg shape: ', code_trg.shape)
    #print('query shape: ', query.shape)
    begDef= time.time()
    with torch.no_grad():
        coordinates = homeomorphism_decoder.forward(code_trg, query)
    endDef= time.time()

    print('coordinates shape: ', coordinates.shape)
    coordinates = coordinates.reshape(B, 50000, 3)

    new_src_mesh_verts_rightSize = [coordinates[s][:num_points]for s in range(B)]
    if(config == "8"):
        new_src_mesh_faces_rightSize = src_mesh_faces_rightSize#.to('cuda')
        #new_src_mesh_faces_rightSize = [changedItem['faces_src'][s][:num_faces[s]].to('cuda') for s in range(B)]
    else:
        new_src_mesh_faces_rightSize = [item['faces_src'][s][:num_faces].to('cuda') for s in range(B)]

    new_src_mesh = Meshes(verts=new_src_mesh_verts_rightSize, faces=new_src_mesh_faces_rightSize)
    end = time.time()

    print('time: ', end-begin)
    print('time def: ', endDef-begDef)

    numberOfSampledPoints=5000
    sample_trg = sample_points_from_meshes(trg_mesh, numberOfSampledPoints).to('cuda')
    new_sample_src = sample_points_from_meshes(new_src_mesh, numberOfSampledPoints).to('cuda')


    


    #loss.backward()

    #scheduler.step()


    #final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    #loss
    for i_item in range(B):
        loss_chamfer, _ = chamfer_distance(sample_trg[i_item].unsqueeze(0), new_sample_src[i_item].unsqueeze(0))
        #loss_chamfer, _ = chamfer_distance(trg_mesh, new_src_mesh)
        loss = loss_chamfer * w_chamfer
        print('loss: ',loss)
        losses.append(loss)

        if(config=="8"):
            item=changedItem
        #else:
        name = item['name'][i_item]
        for key in keys:
            if(key in name):
                ls = chamfer_distance(sample_trg[i_item].unsqueeze(0), new_sample_src[i_item].unsqueeze(0))
                #print(key, ' ', lossIndividus)
                lossIndividus[key].append(ls[0].cpu())
        # scale_trg = item['scale_obj'][i_item].to(device)
        # center_trg = item['center_obj'][i_item].to(device)
        # final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(i_item)
        # final_verts = final_verts * scale_trg + center_trg
        # final_obj = os.path.join(deformed_model+'meshes_valid/', 'mesh_'+name.split('.')[0]+'_'+'.off')
        # save_obj(final_obj, final_verts, final_faces)

        # trg_mesh_draw = trimesh.Trimesh(vertices=final_verts.cpu().numpy(), faces=final_faces.cpu().numpy())
        # trg_mesh_draw.visual.vertex_colors = [0, 255, 0, 50]
        # radii = np.linalg.norm(trg_mesh_draw.vertices - trg_mesh_draw.center_mass, axis=1)
        # trg_mesh_draw.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map='viridis')
        # trg_mesh_draw.export("./temp_obj.obj", include_normals=False, include_color=True)

        # final_verts_trg, final_faces_trg = trg_mesh.to(device).get_mesh_verts_faces(i_item)
        # final_verts_trg = final_verts_trg * scale_trg + center_trg
        # final_obj_trg = os.path.join(deformed_model+'meshes_trg_val/', 'trg_mesh_'+name.split('.')[0]+'_'+'.off')
        # save_obj(final_obj_trg, final_verts_trg, final_faces_trg)


        print(item.keys())
        scale_trg = item['scale_obj'][i_item].to(device)
        center_trg = item['center_obj'][i_item].to(device)
        scale_src = item['scale_src'][i_item].to(device)
        center_src = item['center_src'][i_item].to(device)
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(i_item)
        final_verts = final_verts * scale_trg + center_trg
        if(moreFaces):
            final_obj = os.path.join(deformed_model+'meshes_valid_morefaces/', 'mesh_'+name.split('.')[0]+'_'+'.obj')
        else:
            final_obj = os.path.join(deformed_model+'meshes_valid/', 'mesh_'+name.split('.')[0]+'_'+'.obj')
        #save_obj(final_obj, final_verts, final_faces)

        final_verts_trg, final_faces_trg = trg_mesh.to(device).get_mesh_verts_faces(i_item)
        final_verts_trg = final_verts_trg * scale_trg + center_trg
        final_obj_trg = os.path.join(deformed_model+'meshes_trg_val/', 'trg_mesh_'+name.split('.')[0]+'_'+'.obj')

        final_verts_src, final_faces_src = src_mesh.to(device).get_mesh_verts_faces(i_item)
        final_verts_src = final_verts_src * scale_trg + center_trg
        final_obj_src = os.path.join(deformed_model+'meshes_src_val/', 'src_mesh_'+name.split('_')[0]+'_'+'.obj')
        #save_obj(final_obj_trg, final_verts_trg, final_faces_trg)




        deformed_mesh_draw = trimesh.Trimesh(vertices=final_verts.cpu().numpy(), faces=final_faces.cpu().numpy())
        src_mesh_draw = trimesh.Trimesh(vertices=final_verts_src.cpu().numpy(), faces=final_faces_src.cpu().numpy())
        #print('name: ', name)
        #print(radii[name.split('_')[0]], name.split('_')[0])
        #if(radiiEx[name.split('_')[0]]):
        #print('#################################### name: ', name.split('_')[0])
        colors = np.linalg.norm(src_mesh_draw.vertices - src_mesh_draw.center_mass, axis=1)
        radii[name.split('_')[0]]= trimesh.visual.interpolate(colors, color_map='jet')
        #print(radii[name.split('_')[0]])
        radiiEx[name.split('_')[0]]=False
            
            #done=True

        #deformed_mesh_draw.visual.vertex_colors = [0, 255, 0, 50]
        trg_mesh_draw = trimesh.Trimesh(vertices=final_verts_trg.cpu().numpy(), faces=final_faces_trg.cpu().numpy())
        trg_mesh_draw.visual.vertex_colors = radii[name.split('_')[0]] #trimesh.visual.interpolate(radii[name.split('_')[0]], color_map='viridis')
        trg_mesh_draw.export(final_obj_trg, include_color=False)

        ###########################################################################################3
        
        src_mesh_draw.visual.vertex_colors = radii[name.split('_')[0]] #trimesh.visual.interpolate(radii[name.split('_')[0]], color_map='viridis')
        src_mesh_draw.export(final_obj_src, include_color=False)

        #########################################################################################################################3
        
        #print('mass: ', deformed_mesh_draw.center_mass)
        deformed_mesh_draw.visual.vertex_colors = radii[name.split('_')[0]] #trimesh.visual.interpolate(radii[name.split('_')[0]], color_map='viridis')
        deformed_mesh_draw.export(final_obj,include_color=False)

        
        #trg_mesh_draw.visual.vertex_colors = [0, 255, 0, 50]
        #radii = np.linalg.norm(trg_mesh_draw.vertices - trg_mesh_draw.center_mass, axis=1)

        
            
        pcd = o3d.geometry.PointCloud()
        print('seq_pc_trg[i_item] shape: ', seq_pc_trg[i_item].shape)
        pcd.points = o3d.utility.Vector3dVector(seq_pc_trg[i_item].cpu().detach().numpy()* scale_trg.cpu().detach().numpy() + center_trg.cpu().detach().numpy())
        gt_pcl = os.path.join(deformed_model+'gt_sampled_pcl/', 'pcl_'+name.split('.')[0]+'_'+'.ply')
        o3d.io.write_point_cloud(gt_pcl, pcd)


for key in keys:
    if(len(lossIndividus[key])==0):
        print('does not exist')
    else:    
        print('loss for key: ', key, sum(lossIndividus[key])/len(lossIndividus[key]))

print('len: ', len(losses))
loss_mean = sum(losses) / len(losses)
print('loss_mean: ', loss_mean)
